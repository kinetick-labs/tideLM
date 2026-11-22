# Serving TensorFold's DFlash2 drafter in this engine

TensorFold's production GLM-5.3-Flash on the Spark runs `--drafter /dflash` against
`models/glm53-dflash2`, and its log reports `accepted=62/89`, `95/135`, `61/90` — about 65-70% of
drafted tokens accepted, ~5.5 tokens a verify round. This engine drafts with the checkpoint's own MTP
layer (depth 7, `GLM_MTP_DYN=0.9`), and in `GLM_FLASH_DENSE=exl3` mode the listed-vocabulary draft
head is off. So every decode number compared so far pits a block-diffusion drafter against a stub:
the measurements are not drafter-neutral, and the decode comparison has to be redone once this engine
can draft with the same model.

## What the drafter is

`models/glm53-dflash2/config.json`:

| | |
|---|---|
| architecture | `DFlash2DraftModel` (Qwen3 backbone, `model_type: qwen3`) |
| layers | 5, all `sliding_attention`, `sliding_window: 2048`, `is_causal: false` |
| width | hidden 4096, 32 q heads, 8 kv heads, head_dim 128, intermediate 12288, bf16 |
| block | `block_size: 8`, `mask_token_id: 154856` |
| conv | `conv_kernel_size: 2`, `conv_group_size: 16` — two-tap dynamic convolution |
| selector | `selector_rank: 256`, `selector_top_k: 16` |
| taps | `target_layer_ids: [5, 14, 24, 33, 42]`: the drafter reads the target's hidden states |
| size | 2.2 GB of weights, no KV cache of its own (`use_cache: false`) |

It is a block-diffusion drafter, not an autoregressive one: one pass proposes 8 tokens, keeps the top
candidates at each position, and a rank-256 selector traces one coherent path through them. Because
the target verifies the block, the decoding stays lossless — the receipts are exact greedy outputs,
not approximate ones. That is the property this port has to preserve: **a drafter that is wrong costs
acceptance, never quality.** A lossless check is therefore cheap and mandatory: greedy output with the
drafter on and off must be token-identical.

## Why this engine can take it

* `VERIFY_ROWS = 8` (`rust/crates/tidelm-glm/src/forward.rs`) — the verify machinery already carries
  8 rows a step, and DFlash2's block is 8. No new verify geometry.
* `rust/crates/tidelm-glm/src/inkling_draft.rs` (1 091 lines) is an existing drafter module for
  another family: the shape to follow for a second one (load, plan, propose, hand the block over).
* The engine already has `GLM_MTP` occupying the drafter slot, so the integration point is the same
  one MTP uses (`forward.rs`'s draft chain), behind a new `GLM_DRAFTER=dflash2|mtp|none`.

## TensorFold's implementation, which is the spec

| file | lines | what it is |
|---|---|---|
| `zig/kernels/cuda/glm_dflash.cu` | 354 | taps, per-head RMSNorm + rotate-half RoPE, the two-tap dynamic convs, SwiGLU, ring and block attention (window 2048, 16 key chunks), all bf16 |
| `zig/src/families/glm/dflash.zig` | 369 | layout, buffer sizes, the loader, the selector's candidate layout |
| `zig/glm_dflash_host.zig` | — | host-side tests: gather/read fit at every rank count, scratch sizes for every 4-bit linear |

The kernel's constants are the checkpoint's: `D=4096, HD=128, KVH=8, QH=32, KVW=1024, WIN=2048,
BLOCK=8, GROUP=16, CH=16`, and a per-layer ring of K then V, `[WIN][KVW]` each. TF runs the drafter in
**bf16**, not quantised, so the port has no quantisation step to design.

## Phases

1. **Taps.** In the Rust forward, copy the hidden state after layers 5, 14, 24, 33, 42 into a
   `[5][rows][4096]` bf16 buffer, only when a drafter is enabled. Costs 5 × rows × 8 KB.
2. **Loader.** Read `glm53-dflash2/model.safetensors` into the drafter's buffers with TF's layout
   (`dflash.zig`). 2.2 GB bf16, loaded at boot beside the target's weights.
3. **Drafter kernel.** Port `glm_dflash.cu` into `tools/` beside the other kernels (the same file both
   engines compile), with an FFI entry the Rust drafter module calls: taps → 5 layers → convs →
   selector → 8 candidates per position.
4. **Selector + verify.** Trace the path through the top-16 candidates (rank 256), hand 8 tokens to the
   existing verify, and publish the acceptance counters the server log already reports for MTP.
5. **Measurement.** vs TensorFold's `accepted=62/89`-class numbers on the same prompts, then the
   decode benches (prose, code) with the drafter on, and the drafter-neutral pair (both engines with
   their drafter off) so the engine comparison finally means something.

## Acceptance

* Greedy output token-identical with `GLM_DRAFTER=dflash2` and `none` on 24 fixed prompts — the
  losslessness check, and the first thing to run after any change to the drafter.
* Acceptance rate and tokens-per-round reported on the same 24 prompts; target is TF's 65-70%.
* `--bench-nll corpus/heldout_quick.jsonl` unchanged (1.9250 / 1.6699 / 2.4146): a drafter must not
  move the loss.
* `tools/generation_gate.py` 6/6 and `tools/concurrency_probe.py` aggregate not below 26.26 tok/s.
* Memory: the drafter is 2.2 GB; `MemAvailable` under this engine was 12.4 GiB, so it should land
  near 10 GiB — measure, do not assume.

## The call order, read out of TensorFold

`zig/src/families/glm/dflash_block.zig` is the device sequence and `dflash_select.zig` is the
selector; both are 187 and 64 lines and are the whole specification. Written out, with the buffers
our port names:

**Ingest** (context rows into the rings, `N` rows): `c1 = fc(cin)` where `cin` is `[N][5 * 4096]` (the
five taps) and `fc` is `[4096][20480]`; `ctx = rmsnorm(c1, hidden_norm)`; per layer `ckv = kv(ctx)`
(`kv` is `[2048][4096]`: k and v projected together, `[N][2 * KVW]`), then `df_ctx_kv` norms+rotates
it at `cpos` and writes the ring slot.

**Block** (`M = streams * 8` rows, position `bpos`, ring per stream, `starts`/`froms` per stream),
per layer:

```
y   = rmsnorm(x, in_norm)                 // [M][4096]
dyn = a_proj(y)                           // [M][1024] = [which][tap][group], 2 * 2 * 256
y2  = conv(y, which=0, a_base)            // the two-tap dynamic causal conv, no add
qkv = qkv_lin(y2)                         // [M][6144] = q(4096) | k(1024) | v(1024)
      df_block_qkv → q, kb, vb            // per-head RMSNorm + RoPE at bpos
      df_attn_part + df_attn_merge → att  // sliding window from the ring + the block's own 8 keys
a   = o(att)                              // [M][4096]
x   = x + conv(a, which=1, a_base + 2*D*2) // the same dyn, the second tap pair, added to the residual
y   = rmsnorm(x, post_norm)
dyn = m_proj(y)
y2  = conv(y, which=0, m_base)
gu  = gu_lin(y2)                          // [M][24576]
act = swiglu(gu)                          // [M][12288]
x   = x + conv(down(act), which=1, m_base + 2*D*2)
```

then, once for the block: `hn = rmsnorm(x, norm)` with the 64-column group sums `xs` for a 4-bit head,
`logits = head(hn)` row by row, `df_top16_part` + `df_top16_merge` → 16 candidates a row (id, value),
and `g = hidden_projection(hn)` → `[M][256]`.

**Selector** (rank 0, `dflash_select.zig`): for `t = 1..8`, take the row's 16 candidates, `pg =
predecessor_codebook[prev] * g` (`prev` starts at the anchor), score each candidate
`value + dot(pg, successor_codebook[id])`, keep the argmax, `prev = id`, emit 7 drafts. Greedy, with
the temperature's chance reported beside them. `predecessor_codebook`/`successor_codebook` are
`[154880][256]` bf16 -- the whole vocabulary, which is what makes a 256-wide hidden enough to trace a
path.

Two things this engine has to decide, that TensorFold does not:

* **The head.** TensorFold scores the block with the target's MTP *draft* head (`e.w.mtp.?.draft`,
  4-bit, this rank's vocabulary slice). Our EXL3 boot has no listed-vocabulary draft head (an EXL3
  matrix cannot be gathered by id), so the block's logits have to come from the target's main head
  over 8 rows -- one small matmul a round, and the honest choice until a bf16 head exists.
* **The taps.** Our forward must hand over the same `[rows][4 * 4096]` capture TensorFold's `df_tap`
  reads -- the four streams of the target residual at layers 5, 14, 24, 33, 42, averaged -- or the
  drafter runs on an input it was not trained for and acceptance silently collapses.

## The checkpoint's actual tensors (81, read from the safetensors header)

`fc` 4096x20480, `hidden_norm` 4096, `norm` 4096, `candidate_selector.hidden_projection` 256x4096,
`candidate_selector.{predecessor,successor}_codebook` 154880x256, and per layer (5):
`input_layernorm`, `post_attention_layernorm` 4096; `self_attn.{q_proj}` 4096x4096, `{k_proj,v_proj}`
1024x4096, `o_proj` 4096x4096, `{q_norm,k_norm}` 128; `attention_conv.{base_kernel}` 2x2x4096 and
`{kernel_projection}` 1024x4096; `mlp.{gate_proj,up_proj}` 12288x4096, `down_proj` 4096x12288;
`mlp_conv.{base_kernel}` 2x2x4096 and `{kernel_projection}` 1024x4096. **No `lm_head` and no
`embed_tokens`**: the drafter scores through the target's head and reads the anchor's embedding from
the target.

## Where it plugs in, exactly

The serving path is `tidelm-engine::forward` (the server is on `tidelm_engine`), not
`tidelm-glm`'s Engine; that crate's layer loop is the one to hook. The pieces are done and
compiled -- `dflash.rs` (weights), `dflash_fwd.rs` (`Run`, `Block`), `dflash.cu` (16 launchers) --
so the wiring is the remaining work, and it is three edits plus one module:

1. **Taps.** In the layer loop, after layer `l`'s residual write, when the drafter is on and
   `l` is one of `[5, 14, 24, 33, 42]`: `run.tap(st, residual4, slot)`. `residual4` is the
   mHC path's `[rows][4][D]` buffer -- the same one `df_tap` averages, so no repacking.
2. **A round.** `tap x5 -> embed_taps -> pass -> the target's head over the 8 rows into the
   engine's fp32 logits -> candidates -> anchor -> walk -> drafts`. The head stays the
   target's: the checkpoint has no `lm_head`, and scoring the block through the drafter's
   own head would be a different distribution than the one it was trained against.
3. **The driver.** Mirror `Dspark` in the same file (`dspark: Option<Dspark>`, `dspark_seed`,
   `dspark_draft`, `opts.spec`): a `Dflash` variant with `seed` (the prompt's last `WIN`
   tokens through `ctx_kv` at each layer, into the ring) and `draft` (one round, `BLOCK - 1`
   ids), handed to the existing block verify (`tidelm_core::spec::verify_sparse`), which is
   already lossless for greedy. `GLM_DRAFTER=dflash2` selects it; unset keeps today's path.

The first test is not acceptance, it is identity: drafter on and off must emit the same greedy
tokens on 24 prompts. Acceptance (TensorFold measures ~5,6 tokens a pass) comes after, and only
then the throughput comparison, which today is still not drafter-neutral.

## The tap is *after* the layer, and the engine's existing hook is *before*

`zig/src/families/glm/cuda_taps.zig` settles it, and it is the kind of detail that fails
silently:

    //! DFlash2's taps on the target's forward: after each tapped layer, its rows' 4-stream mean.
    pub const layers = [_]usize{ 5, 14, 24, 33, 42 };
    /// After layer `j` ... rows lo..lo+n of residual `x` (bf16 [rows][4][hidden]).

TensorFold's `Taps.at(j, x, lo, n)` runs after layer `j`'s residual is written. The engine has a
near-identical hook for DSpark, at `forward.rs:3863`, and it runs **before** `self.block(l)` --
"the DSpark main hidden reads the attention INPUT of its target layers", i.e. the residual after
layer `l - 1`.

So `dspark_target_layer_ids` cannot be reused for DFlash2 even though it is the same measurement
into the same shape: pointing it at `[5, 14, 24, 33, 42]` would feed the drafter the residuals of
layers 4, 13, 23, 32 and 41. Nothing would error. The acceptance rate would simply come out worse
than it should, and the obvious explanation would be the obvious wrong one.

The drafter therefore gets its own hook after `self.block(...)`, its own buffer, and the same
`df_tap` mode 0 (the 4 streams' mean) -- which is the same buffer layout `main_h` already has
(`[rows][5][D]`), so `ds_gather_rows_u16` gathers the block's 8 rows at draft time unchanged.

## Open: the block pass is not deterministic yet

Three runs of `glm_bench_dflash --load ... --rows 8` on the same checkpoint and the same
deterministic taps:

    hn: 32768/32768 finite, 6 exact zeros, range [-10.1250, 6.8750]
    hn: 32768/32768 finite, 2 exact zeros, range [-9.7500, 13.0000]
    hn: 32768/32768 finite, 3 exact zeros, range [-10.6250, 9.5000]

Same inputs, three different blocks. Something reads memory that was never written, or a race in
one of the six launches a layer. The candidates, in the order I would eliminate them:

- `attn_part`'s chunk count: `Run::pass` passes `CH` (16) as the number of key chunks, which is
  the only argument whose meaning I inferred rather than read out of the kernel. If the kernel
  derives its slice from it, the partials it never writes are read back by `attn_merge` from
  whatever the previous call left -- and on a fresh allocation, from nothing.
- The empty context: with `starts = froms = 0` the ring holds no keys, and the window arithmetic
  at the block boundary (`from` before row 0) is a case TensorFold may never exercise in serving.
- `pm`/`pl`/`pacc` between launches.

This has to be settled before any acceptance number: a drafter that scores a different block each
run cannot be reasoned about, and the acceptance rate would look plausibly bad for a reason that
is not the design.

### What the bisection says

`glm_bench_dflash --layers N` runs the first N layers, three passes in one process:

- **N = 0** (fc + hidden_norm + final norm + selector projection, no layer at all): pass 0 differs,
  passes 1 and 2 agree exactly. `hn[0..4]` = [0.016, 0.041, 0.020, 0.087] against
  [0.81, -0.77, -3.38, 0.67]. The input is not the variable -- `cin` read back from the device has
  the same 163840 values and the same hash every pass.
- **N >= 1**: every pass differs, including two passes run back to back.

So there are two faults, not one. The first is a cold-start read (the first execution after the
warm-up sees something the later ones do not, with no layer involved, which puts it in `embed_taps`
or the final norm). The second is per-call drift as soon as a layer runs, which is what makes the
acceptance measurement meaningless until it is fixed.

Neither is a numerical-noise story: a bf16 rounding difference cannot turn a sum of 32768 hiddens
from +30 to -85, and cannot move `hn[0]` from 0.016 to 0.81.

## Settled: one null stream, not two faults

The block pass was nondeterministic because `embed_taps` passed `null_mut()` as the stream to
`glm_rmsnorm` (`dflash_fwd.rs`, the hidden norm). That launch went to the **default stream** while
the `fc` GEMM before it ran on the ring's stream: nothing ordered the two, so the norm read the
gemm's output while it was still being written.

It looked like two faults and was one. The bisection read as "cold start plus per-call drift"
because a race is a function of timing: at `--iters 1` the third pass already matched, at
`--iters 3` none of them did, and the value all runs converged on (sum -84.817) was the race
losing every time.

With `st.raw()` there, three passes and two separate processes give the same block, bit for bit:

    min -14.8750 max 10.5625 sum -90.274 first [-0.3203, 0.5586, 0.4707, 2.4844]

`DF_DUMP=1` is what made it visible: the first label (`l0 norm`) already disagreed, which put the
fault upstream of every layer kernel rather than inside one. The instrumentation stays -- the next
noon one is inside `embed_taps` or the head, and it costs nothing when the variable is unset.

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

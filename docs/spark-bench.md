# What to run on the Spark

Nothing in this change was executed on a GPU. The machine that wrote it has rustc 1.83 and no
nvcc, so `cargo test -p tidelm-dflash` cannot build (`tidelm-kernels/build.rs` compiles
`cuda/dflash.cu`). The host geometry is tested; the kernels, the head, and every tok/s number
are not.

Verified here, on the CPU:

* `rustc --test rust/crates/tidelm-dflash/src/plan.rs` — 5 tests. The block is the anchor then
  seven masks. RoPE's position and attention's `starts + t` are the same number. A window that
  crosses a multiple of 2048 is two slices. A prefill or a rewind drops the ring; a full accept
  (anchor advances by 8) keeps it.

Not verified:

* That the server builds (`scripts/tide.sh build`).
* That a round accepts more than the anchor. The campaign's 1.04 is the number to beat, and it
  was measured with the old block. This tree has not been served.
* Prefill. No kernel in `tools/exl3_moe_cuda.cu` changed. The comparison that says the fat tile
  is already the same shape as TensorFold's `tgateup` is a reading of the two sources
  (`docs/exl3-prompt-port.md`), not a timing.
* Memory. The drafter's weights are the published 2.2 GB, allocated only when `GLM_DRAFTER` is
  set. The new ingest scratch is one window: two `[2048][4096]` bf16 buffers and one
  `[2048][2048]` (about 40 MB). The capture ring (84 MB) was already there.

## Boot

Copy the profile and point it at the two checkpoints. The drafter lines are commented in
`configs/glm53-flash-248-exl3.env.example`; uncomment them. `GLM_MTP_DEPTH` stays 7: Flash has
an MTP layer, and the depth the block is allowed to propose is that setting. Attaching the
drafter stops the MTP chain (one drafter a lane).

```bash
cp configs/glm53-flash-248-exl3.env.example .env.glm53-dflash
# edit GLM_MODEL_DIR, GLM_EXL3_DIR, and:
#   GLM_DRAFTER=dflash2
#   GLM_DRAFTER_DIR=/home/you/models/glm53-dflash2
#   GLM_COPY=0
#   GLM_LOOKUP=0
#   GLM_DFLASH_LOG=1
scripts/tide.sh build .env.glm53-dflash
scripts/tide.sh start .env.glm53-dflash
```

In another shell, while it boots and while the first bench runs:

```bash
tools/mem_watch.sh 300 1 > results/mem-dflash.csv
```

The boot line to look for is `boot: DFlash2 drafter`. It names the mask (154856 unless the
drafter's `config.json` says otherwise). A boot that says the checkpoint is not Flash, or that
the world is not 1, is the guard doing what it should: two ranks would draft from different
head shards and not raise.

`tools/mem_watch.sh` prints the minimum MemAvailable on stderr. The previous tide boot, without
this drafter, left 12.4 GiB at 9700 slots. TensorFold's GLM branch left about 5 GB, sometimes
cited 3.7–3.9 GiB. The drafter should cost about 2.2 GB plus the scratch, and the free memory
should stay well above TensorFold's. If it does not, stop and do not raise `GLM_SLOTS`.

## Acceptance, before any tok/s claim

One request at a time. Two drafting lanes share one ring and the round drafts none of them
(`dflash: N lanes in one round`).

Temperature 0 is the measurement. `bench.py` defaults to 0.6, and rejection sampling accepts
less than greedy. Copy and lookup must be off in the profile: a copy hit never calls the
drafter, and `accept_len_mean` would count it.

```bash
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload prose --osl 256 --temperature 0 --runs 1 --warmup 0 \
    --out results/dflash-prose-t0.json
scripts/tide.sh logs -n 200 .env.glm53-dflash | grep '^dflash:'
```

The first two rounds always print, and every round prints while `GLM_DFLASH_LOG=1`:

```text
dflash: pos 3000 anchor 12345 attn [952, 3000) ring_was 0, drafts [..7 ids..], row1 [..16 ids..]
```

* `row1` values in `0..15` mean the head is still dead. Stop. The block was scored on zeros.
* `row1` values in the tens of thousands mean `lm_head(hn)` ran. Then `accept_len_mean` is the
  number. The anchor counts as 1. The campaign's DFlash-alone figure is 1.04. TensorFold's
  reported verifies are about 5.5 tokens (accepted 62/89, 95/135, 61/90 on its own log).
* `drafting off for this process` after 8 rounds with no accepted draft is the stand-down. The
  verify was costing 8 rows to return 1 token (measured 5.17 against 19.38 tok/s before this
  rewrite). A new process brings it back. Do not read a tok/s from a run that printed that line.

Then the same prompt at the temperature the profile's prose number used, and the code prompt:

```bash
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload prose --osl 1024 --temperature 0.6 --runs 3 --out results/dflash-prose.json
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload code --osl 1024 --temperature 0.6 --runs 3 --out results/dflash-code.json
```

The profile without the drafter, same commands, is the decode baseline: 33.43 prose / 32.68
code tok/s at the shipped MTP settings, or 27.11 / 35.04 at temperature 0.6 on the native EXL3
boot (`configs/glm53-flash-248-exl3.env.example`). TensorFold's decode on this box is 26.4
prose / 36.7 code. Compare against those, and write the number next to the workload. A run
that stood down is not a DFlash number.

`tools/generation_gate.py` is a floor, not a quality score. It has passed configurations that
emit Python which does not parse. Run it anyway, at the drafter's boot:

```bash
python3 tools/generation_gate.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --max-tokens 1200 --json results/dflash-gate.json
```

## Prefill

Unchanged by this patch, so a regression here is a build or a config difference, not the
drafter. Prefill tok/s is `prompt_tokens_actual / (ttft_ms / 1000)` from the JSON. The two
lengths are the ones in `docs/exl3-prompt-port.md`.

```bash
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload random --isl 6830 --osl 64 --temperature 0 --ignore-eos --runs 3 \
    --out results/prefill-6830.json
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload random --isl 27290 --osl 64 --temperature 0 --ignore-eos --runs 3 \
    --out results/prefill-27290.json
```

TensorFold on these lengths: 7.58 s (901 tok/s) and 21.5 s (1269 tok/s). TideLM's last
measurements: about 9.7 s (702–769) and 36 s (752). The fat tile already matches `tgateup`'s
shape (256 threads, 64 members, gate and up together, fused Hadamard and SwiGLU, 128 columns
a block). What is not the same is the pipeline depth, 3 against 4, and which warps own gate.
That is not a reason to retile until a profile says the expert prompt is the 1.7×. The
length scaling is the symptom worth timing: TensorFold's tok/s rises from 6.8k to 27k tokens
and TideLM's does not.

NLL, if a tile does change later: `--bench-nll` on `corpus/heldout_quick.jsonl`, and the floor
to hold is 1.9250 / 1.6699 / 2.4146. A fourth-decimal move is the accumulation order. A
third-decimal move is a wrong port. Not this change.

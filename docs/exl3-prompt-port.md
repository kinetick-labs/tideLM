# Where the prefill gap is, and what the port has to reproduce

Measured, one Spark, GLM-5.3-Flash SAGE-REXL3 2.48bpw, 1 024-token generations after the prompt:

| prompt tokens | TensorFold (Zig) | TideLM (Rust), `GLM_CHUNK=4096` |
|---|---|---|
| 6 830 | 7.58 s (901 tok/s) | 9.72 s (702 tok/s) |
| 27 290 | 21.5 s (1 269 tok/s) | 36.3 s (752 tok/s) |

`GLM_CHUNK` 8192 is no better than 4096, and `GLM_PREFILL_TWO_PASS=1` is worse (694 / 748). Decode is
not the question: prose and code sit within ±5% of each other (27.11 / 35.04 against 26.44 / 36.69).

## Neither engine is FP8 here

Checked in the sources, because it decides whether the port can be quality-neutral:

* TensorFold's routed-expert prompt kernel is `tf_exl3_prompt_k<K2>` → `pg_program<2, K2>`
  (`zig/kernels/cuda/fn_exl3.cu`), and its MMA is `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
  — **fp16**. `exl3_prompt_x8.cuh` (e4m3, `m16n8k32`) exists but serves the candidate arm
  (`exl3_candidate.cu`), not the GLM prefill that produced the numbers above.
* TideLM's `prefill_fat_kernel` (`tools/exl3_moe_cuda.cu`) is also **fp16** `m16n8k16`, with the
  trellis decoded in-kernel, 4 pipeline stages, 256 threads, 64 member rows × 128-column blocks,
  SwiGLU epilogue fused. `TIDELM_GLM_FAT` is on by default (`forward.rs`: `!= Ok("0")`), and turning
  it off measured 18.29 → 15.82 s.

So the 1.7× at 27K tokens is **implementation, not dtype**: the same fp16 MMA, arranged differently.
That is the good news — a port of the tiling has no numerical excuse to fail, and it must still be
proved with `--bench-nll` (1.9250 / 1.6699 coding / 2.4146 general on `heldout_quick`, the number to
hold) and `tools/generation_gate.py` (6/6).

## What TensorFold's prompt arm does differently

`pg_program<2, K2>` with a 128-thread block (4 warps), fed by a separate 1024-thread item list
(`tf_exl3_prompt_items`): per expert, members are grouped in 64s, and the grid walks (item, column
tile) so a block owns one 64-member group of one expert over one column tile. Against TideLM's fat
tile:

| | TensorFold `pg_program` | TideLM `prefill_fat_kernel` |
|---|---|---|
| block | 128 threads, 4 warps | 256 threads, 8 warps |
| rows per block | one 64-member group | `FAT_MB * 16` = 64 members |
| columns per block | one tile, grid.x walks them | 128-column block, `NS` streams (2: gate, up) |
| gate / up | one matrix per launch | both at once, `NS=2`, epilogue fused |
| staging | cp.async, `TSTAGES=3`-style ring | `FAT_STAGES=4` |
| routing input | item list from its own kernel | `uids` / `ucount` / `members` |

The plausible wins are the warp-to-tile mapping and what a block re-reads: TideLM's `NS=2` keeps two
streams in flight per block and spends registers on `acc[2][4][2][4]`, while TensorFold keeps one
matrix per launch and a narrower accumulator. Which of the two matters is not something to argue:
`--bench-prefill-ab` (see `forward.rs`, the switch that measured 18.29 → 15.82 s) exists to answer it
on the real checkpoint.

## Order of work

1. A `GLM_PREFILL_KERNEL=tfpg` arm beside the fat tile — a `prefill_pg_kernel` with TensorFold's
   mapping (4 warps, one 64-member group, one column tile per block, gate and up as separate
   launches), decoding the trellis exactly as `fat_tiles` does today. Same fp16 MMA, same codebook,
   same epilogue arithmetic; only the arrangement changes.
2. A/B with `--bench-prefill-ab` at 8K and 32K, then `probe_prefill.py` for 8K / 32K / 269K, three
   runs each, median.
3. `--bench-nll corpus/heldout_quick.jsonl` must stay at 1.9250 / 1.6699 / 2.4146. A tiling change
   reorders fp32 accumulation, so a difference in the fourth decimal is expected and acceptable; a
   difference in the third is not, and means the port is wrong, not faster.
4. `tools/generation_gate.py` 6/6, and `tools/concurrency_probe.py` unchanged (the port must not cost
   the concurrent case its 26.26 tok/s aggregate).
5. Only if that lands: e4m3 (`m16n8k32`) as a second arm, with `tools/exl3_prefill_ref.py` bounding
   what the quantisation costs before any adoption. Not before: it moves values, and the operator's
   rule is that speed may not be bought with quality.

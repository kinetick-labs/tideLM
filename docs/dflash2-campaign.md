# DFlash2 on GLM-5.3-Flash: what the campaign measured, and what it concluded

This is the record of a drafter integration — nine commits, one working kernel set, one
end-to-end serving path, and a negative result that is more useful than the positive one
would have been. It is written for the person who finds this in six months and wonders
whether the drafter is worth picking up.

## The columns, all on one box, one prompt, temperature 0

| configuration | decode | tokens a step | notes |
|---|---|---|---|
| no drafter | 19,38 tok/s | 1,00 | `--depth 0` |
| MTP chain, `--depth 7`, the config the campaign started from | 29,82 tok/s | 3,85 | `--dyn 0 --hot 0 --lookup 0`, no `--graphs` |
| MTP chain, **the shipped profile** | **33,43 prose / 32,68 code** | 2,95 / 2,76 | `--cal 0.36 --graphs --hot 1 --hot-async 1 --lookup 8` |
| DFlash2, 8-key seed | 8,89 tok/s | 1,64 | |
| DFlash2, whole-window seed | 7,00–8,04 tok/s | 1,48–1,70 | 15 and 804-token prompts |
| DFlash2 alone, MTP off on the lane | 18,55 tok/s | **1,04** | the honest number for the block |

TensorFold, same box, same model: 26,44 prose / 36,69 code decode, 1 269 tok/s prefill
(27 290 tokens in 21,5 s, `docs/exl3-prompt-port.md:8`). The 1 923 tok/s figure that appears
in older notes is not a prefill measurement on this prompt and should not be used.

So the engine is at parity on decode in prose (+26 %) and 11 % behind in code, and the
remaining gap is prefill: 768–769 tok/s against 1 269.

## What the drafter needed, in the order it was found

1. **Two out-of-bounds reads** in `glm_df_ctx_kv`: `rings` indexed by the row instead of the
   stream (8 bytes past an 8-byte allocation), and `pos` read as `long long` over an i32
   buffer. compute-sanitizer found them; the bench could not, because it never launched the
   kernels that faulted (`47b5416`).
2. **The scoring kernel read the router's buffer**, not the logits: `act.logits` is
   `mr.max(d) * experts`, the target's head leaves `act.head` (`[HEAD_ROWS][vl]`, f32). The
   top-16 now takes `act.head` through the f32 variant (`47b5416`).
3. **`chunks` came from the checkpoint's VOCAB (38)** while the kernel grids over the
   target's head width (147 456 → 36): the merge read 32 never-written slots a row, value
   0.0 with id 0, and 0.0 beats every negative logit (`b61cab6`).
4. **Rows were rotated a token** against their own content: the block declared row `i` at
   `p0+i` while the taps hold `p0-1+i` (`b61cab6`).
5. **The seed wrote outside the window attention reads**: at `p0..p0+7` while attention
   reads `[from, p0)`, so two thousand zeroed slots drowned the softmax (`b61cab6`).
6. **The tap fired on two passes out of eleven.** Its condition was `t == VERIFY_ROWS` (8),
   but a verify pass packs one block a lane (`row0 += 1 + nd`), so its width was 15. Tapping
   rows 0..8 of any pass between 8 and 16 rows wide moved acceptance for the first time,
   from 1,03 to 1,64 (`9198bf9`, `1965ad5`).
7. **The round cost 190 ms because of allocations**, not the two stream joins: `set_tables`
   and `anchor` allocated and freed ~1 MiB a call, five a round, and `cudaFree` synchronizes
   the device. Uploading straight into preallocated buffers removed that (`b61cab6`). Removing
   the joins measured no difference and made `glm_df_select` fault, so they stay.

## What the campaign concluded, and what was wrong with it

**Context is not the limit.** The drafter was seeded from eight keys, then from the whole
captured window (80 keys, then 800). Eight keys and eight hundred keys accept the same 0,6
draft a round. That comparison does not survive: both seeds were scored against the previous
forward's logits, so the window never had a chance to matter. The two commits that claimed
the window was the fix were wrong (`cf69b12`), and so is this paragraph's conclusion.

**The lane was running two drafters.** `drafts` held fourteen entries — the block's seven,
then the MTP chain's seven — and the acceptance loop stops at the first mismatch. With the
MTP chain stopped on that lane, the honest number appears: **accept 1,04** (`ff5d1a6`).
That number is real. The explanation under it is not.

**The drafts `8, 0, 13, 4, 8, 0, 0` are token ids, of a dead distribution.** The top-16
writes absolute ids (`si[j] = offset + c0 + j`, then the merge copies the winning id). A row
of zeros, or an unwritten row, tie-breaks to the lowest ids, and a 1-row decode only writes
row 0 of `act.head` (`HEAD_ROWS` is 33). The block was scored on that buffer. The walk was
not returning ranks. Vocab on this checkpoint is 154880, not 147456; 36 chunks was
36 × 4096, which is neither.

## What TensorFold actually does (read from `rebase/1.3.1-glm53`, not from the notes above)

`Drafter.draft` (`zig/src/families/glm/dflash.zig`) and `pass` (`dflash_block.zig`):

1. The block's row 0 is the embedding of the anchor. Rows 1..7 are the embedding of the mask
   token (154856). The residual is not `rmsnorm(fc(taps))`. The taps go through `ingestPass`
   (`fc`, hidden norm, each layer's `kv`) into the ring, at the row's real position, and only
   for positions the ring does not already hold.
2. The head is applied to the drafter's own `hn` (`rmsnorm(x, norm)` after the five layers),
   not to the previous target forward. On the CUDA path that head is `e.w.mtp.?.draft`, a
   4-bit quant. This tree has no gatherable draft head in EXL3 mode, so the block is scored
   with the target's EXL3 `lm_head` over the eight rows of `hn`. Same shape (4096 → 154880),
   the same call `head_dev` already makes for a verify. If acceptance is still ~1 after this
   and the logged candidates are real token ids, the two heads are the remaining difference.
3. Attention's position is `starts + t` with `starts` equal to the anchor's position, and
   RoPE uses the same numbers (`bpos[t] = anchor + t`). The port had set `starts` one past
   `bpos`. The anchor occupies `lane_pos` at the start of the round: it has been sampled and
   not yet written, and the verify forward begins there.
4. The tap is the 4-stream mean **after** the layer's MLP site is mixed back into the
   residual (`cuda_forward.zig`, `t.at` after `post`), layers 5, 14, 24, 33, 42. Tapping
   `act.h` after `mlp()` reads the post-attention stream; the MLP residual is still in
   `act.a`. And `--graphs` never entered `layers()`, so a verify on the shipped profile did
   not tap at all.
5. The selector multiplies `g` as fp32 (`linB(..., wide: true)`). A bf16 store quantises the
   256-vector before the dot. That is a close-call change, not the 1,04.

`seed_window` stays in the tree and is not called. It ran a block pass and then `ctx_kv` on
the last layer's `qkv`, which is not `ingestPass`. The 1,48–1,70 it measured was this bug
plus the dead head.

The server now does, in order: ingest the gap in the capture (one GEMM a contiguous slice of
the window, not 256 passes of 8), embed `[anchor, mask × 7]` with `pos` null so position 0
is a real row, run the five layers, score `hn` with `lm_head`, walk. The first two rounds
print `dflash: pos … drafts … row1 …`. `GLM_DFLASH_LOG=1` prints every round. Row 1's
sixteen ids are the check the campaign asked for: values past a few dozen mean the head is
alive; `0..15` means it is not, and the rest of this change did not land.

One capture and one ring. Two drafting lanes in a round draft none of them (they would alias
`p % WIN`). A decode round advances the anchor by at most one block; a prefill or a rewind
re-ingests the window, because the ring would still be the previous span.

No acceptance number in this section was measured on a GPU. The 1,04 stands until the Spark
runs the commands in `docs/spark-bench.md`.

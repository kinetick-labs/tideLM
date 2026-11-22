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

## What it does not need, and this is the part that matters

**Context is not the limit.** The drafter was seeded from eight keys, then from the whole
captured window (80 keys, then 800), through a capture that writes the last WIN rows'
residuals at the five tapped layers during prefill — 84 MB, and a seed pass that reuses the
lane's own ring so it allocates nothing. Eight keys and eight hundred keys accept the same
0,6 draft a round. The two commits that claimed otherwise were wrong (`cf69b12`).

**The lane was running two drafters.** `drafts` held fourteen entries — the block's seven,
then the MTP chain's seven — and the acceptance loop stops at the first mismatch, so the
block's seven were the only ones ever compared. With the MTP chain stopped on that lane, the
honest number appears: **accept 1,04, i.e. nothing, every single time** (`ff5d1a6`).

**And the drafts are not tokens.** They come out as `8, 0, 13, 4, 8, 0, 0` — eleven-bit ids
where a token id runs to 147 455. The top-16 keeps absolute ids (`si[j] = offset + c0 + j`)
and the merge copies them through, so the suspect is the walk: `select` reads candidates,
looks the predecessor up in `pred` and the successor in `succ`, and something in that path
hands back rank-sized numbers.

## Conclusion

The integration is real: the block loads, runs, serves, and its cost is measured. The
drafter is not: **it has never produced a single accepted token**, and every earlier
explanation of that (the window, the positions, the chunks, the synchronizations) was
downstream of the one defect still open — the candidates or the walk handing back indices
instead of tokens.

The next step is one build and one request: print the sixteen candidates a row carries before
`glm_df_select` runs. Token ids mean the walk is wrong; 0..15 mean the top-16 writes ranks
into the id slot, and every draft the campaign produced was noise.

Until that is answered, the drafter should stay off in production, which is where it is:
`GLM_DRAFTER` unset. The MTP chain at the shipped profile is the configuration to serve, and
prefill — not the drafter — is where the remaining 30 % lives.

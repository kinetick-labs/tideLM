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

## Addendum: the open item was the wrong question

The sixteen candidates were not the defect. `glm_df_select` writes `cand[r * 32 + best]`, the
token id, and the top-16 stores absolute ids. `8, 0, 13, 4, 8, 0, 0` are legal token ids. They
are small because the distribution they were drawn from was not the drafter's. Printing them
(`DF_CAND=1`, the first drafted row) is still worth one request on the Spark: token-scale ids
confirm this, and `0..15` would mean the kernel was changed out from under the comment.

What the round actually computed, against z-lab's `dflash_generate` and TensorFold's
`dflash.zig` / `dflash_block.zig` on the same checkpoint:

1. **The block input was `rmsnorm(fc(taps))`.** The trained block is the target's embedding of
   the anchor and seven copies of mask token 154856. `round_bf16` called `embed_taps` and then
   `pass`.
2. **The logits were the previous verify's `act.head`.** The head has to run on the drafter's
   own `hn` after `pass`. The checkpoint has no `lm_head`; the target's head is the one it was
   trained against. TensorFold quantises that same matrix to 4 bits. This tree scores the EXL3
   head, which is the serving head, over the eight rows.
3. **The context was not `kv(rmsnorm(fc(taps)))`.** `seed_taps` projects each tap through that
   layer's `k|v` separately. `seed_window` runs the block pass and then `ctx_kv` on `qkv`, and
   `qkv` is Q|K|V, so the 2048 columns the kernel reads are Q. The reference writes K and V of
   the fused taps, at the context positions, and does not run the block to do it.
4. **The residual was captured a site early, and not on the path that serves.** On the mHC
   model the MLP is mixed into the residual at the *next* layer's `hc_post` / `hc_site`
   (`flash_fwd.rs`, the dump labelled `x{l-1}`). The tap sat after `mlp()`, so it read the
   stream before that mix. Decode with graphs replays `decode_segment`, which never called the
   tap, and a verify is at most 16 rows so it takes that path. The prefill window tap did run,
   on the pre-MLP residual, and only when `t > 16`.
5. **Attention's start was one past the RoPE positions.** `p0 = lane_pos + 1`, `bpos = p0 - 1
   + i` (which is `lane_pos + i`, the right RoPE position) and `starts = p0`. The attention
   kernel uses `starts` and ignores `bpos` for the window. The anchor is consumed at
   `lane_pos`.

Any one of (1), (2) or (3) is enough to leave acceptance at the anchor. The "context is not
the limit" column (eight keys and eight hundred keys, both ~0.6 of a draft) was measured with
(1) and (2) in place, so it does not say the window is irrelevant once the block is embeddings.
The position rotation in `b61cab6` fixed the RoPE half of (5) and left the attention half.

A verify also taps the drafts it then rejects. Those rows share a window slot with a position
2048 back, so the slot no longer holds that position. Within one request the ring is only
appended to, and the rejected positions are past the next anchor, so they are not re-read. A
new request on the lane is different: `start` rewinds the target cache to `keep` and the
drafter forgets the ring. The fold and the attention both start at `keep`, because the slots
below it can be that abandoned tail. A fresh prompt (`keep` 0) still folds the whole window.
A follow-up whose suffix is shorter than 2048 drafts against the suffix until a window of new
tokens has been tapped.

The selector walk was left alone. It already drops row 0 and chains the predecessor from the
anchor, which is what `CandidateSelector.select` does.

What this costs on top of the boot that left 12.4 GiB free: the drafter's 2.2 GiB, plus a lane's
ring and window (40 MB + 80 MB) and the block workspace the bench already allocated (~120 MB).
At `GLM_LANES=2` that is about 2.6 GiB. The expected free memory is near 10 GiB, still ahead of
TensorFold's 3.7–3.9 GiB. It is not turned on in the profile. Nothing here was run on a GPU;
the acceptance number is still 1.04 until the Spark says otherwise. TensorFold's published
band on this model is about 5.5 tokens a round and roughly 65–70% of the drafted tokens. That
is the band the commands below are aimed at, not a result.

## What to run on the Spark

Greedy output with the drafter on and off has to be the same tokens. A wrong draft is rejected
by the verify, and the verify is the same forward. The draft's head write lands in `act.head`
and the verify overwrites it before acceptance reads the argmax. If the tokens differ, the
verify moved, and that is a bug in this change.

```
cp configs/glm53-flash-248-exl3.env.example .env.glm53-flash-248
# edit the paths, then add:
#   GLM_DRAFTER=dflash2
#   GLM_DRAFTER_DIR=/home/you/models/glm53-dflash2
#   GLM_COPY=0
#   GLM_LOOKUP=0
# leave GLM_MTP=1 and GLM_MTP_DEPTH=7
scripts/tide.sh start .env.glm53-flash-248
# the boot line must contain "boot: DFlash2 drafter"
# the first request prints one line: dflash: lane N anchor A at P, context [lo, hi), drafts [...]
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload prose --temperature 0 --runs 3 --out results/dflash-prose.json
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload code --temperature 0 --runs 3 --out results/dflash-code.json
python3 tools/generation_gate.py --base http://127.0.0.1:8000 --model glm-5.3-flash
python3 tools/concurrency_probe.py --base http://127.0.0.1:8000 --model glm-5.3-flash
tools/mem_watch.sh 300
```

The request line is `drafted / accepted a round`. Tokens a step is one more than accepted,
because the anchor is always emitted. The floor from the MTP profile stays: concurrency's
aggregate not below 26.26 tok/s, and the gate 6/6. `DF_CAND=1` on one short request prints
the sixteen candidates of the first drafted row.

Prefill is unchanged by this and is still the open 30%. The tile port
(`GLM_PREFILL_KERNEL=tfpg` in docs/exl3-prompt-port.md) is not in this tree: it cannot be
compiled or measured here, and a default-on tiling change is a numeric change the pair would
not report. The commands, when that kernel exists:

```
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload random --isl 8192 --osl 128 --runs 3
python3 bench/bench.py --base http://127.0.0.1:8000 --model glm-5.3-flash \
    --workload random --isl 27290 --osl 128 --runs 1
# then the same two with the flag off. Hold NLL at 1.9250 / 1.6699 / 2.4146
# (corpus/heldout_quick.jsonl, a third-decimal move means the port is wrong).
```

The reference prefill is 6,830 tokens in 7.58 s (901 tok/s) and 27,290 in 21.5 s (1,269).
TideLM at `GLM_CHUNK=4096` is 9.72 s (702) and 36.3 s (752). `GLM_CHUNK=8192` was no better
and `GLM_PREFILL_TWO_PASS=1` was worse (694 / 748).

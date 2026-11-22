//! Host-side geometry of one DFlash2 round.
//!
//! TensorFold's `Drafter.draft` (`zig/src/families/glm/dflash.zig` on `rebase/1.3.1-glm53`) is the
//! spec, and it is small enough to test without a GPU:
//!
//! * the block's row 0 is the anchor token at `anchor_pos`; rows 1..7 are the mask token at
//!   `anchor_pos + t`. Attention's position for row `t` is `starts + t` with `starts = anchor_pos`,
//!   so the two have to be the same number or RoPE and the window disagree.
//! * the ring holds positions `[max(0, anchor_pos - WIN), anchor_pos)`. The anchor itself is a
//!   block row, not a context row.
//! * the capture buffer is a ring of `WIN` slots (`p % WIN`). A window that crosses a multiple of
//!   `WIN` is two contiguous slices.

/// Tokens a block proposes. Must stay equal to `tidelm_kernels::dflash::BLOCK` (checked at load).
pub const BLOCK: usize = 8;
/// Context keys the ring holds. Must stay equal to `tidelm_kernels::dflash::WIN`.
pub const WIN: usize = 2048;
/// `mask_token_id` in `models/glm53-dflash2/config.json`. A checkpoint that publishes another one
/// overrides this at load.
pub const MASK: i32 = 154856;

/// One stream's token ids: the anchor, then the mask. Row 0's hidden predicts nothing the walk
/// reads; rows 1..7 are the drafts, in order.
pub fn block_ids(anchor: i32, mask: i32) -> [i32; BLOCK] {
    let mut ids = [mask; BLOCK];
    ids[0] = anchor;
    ids
}

/// RoPE position of each block row. Attention uses the same numbers when `starts` is `anchor_pos`.
pub fn block_positions(anchor_pos: i64) -> [i64; BLOCK] {
    core::array::from_fn(|t| anchor_pos + t as i64)
}

/// The position attention computes for block row `t` (`glm_df_attn_part`: `p = starts + j`).
pub fn attention_position(starts: i64, row: usize) -> i64 {
    starts + row as i64
}

/// Oldest position the ring can still hold when the anchor sits at `anchor_pos`.
pub fn context_from(anchor_pos: i64) -> i64 {
    (anchor_pos - WIN as i64).max(0)
}

/// What the ring still holds when the next anchor is `anchor_pos`. A decode round moves the anchor
/// by at most one block (the anchor plus seven drafts). A prefill or a rewind does not, and the
/// ring would still be the previous span.
pub fn resume_until(until: i64, anchor_pos: i64) -> i64 {
    if anchor_pos < until || anchor_pos > until + BLOCK as i64 { 0 } else { until }
}

/// First position still to ingest. `until` is the exclusive end of what the ring already has;
/// anything older than the window was overwritten in the capture and must not be read back.
pub fn ingest_lo(until: i64, anchor_pos: i64) -> i64 {
    let from = context_from(anchor_pos);
    if until < from { from } else { until.min(anchor_pos) }
}

/// One contiguous run of the capture: `n` positions starting at `pos0`, stored at slot `phys`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WinSlice {
    pub phys: usize,
    pub pos0: i64,
    pub n: usize,
}

/// `[start, end)` as one or two slices of the `WIN`-slot capture. `end - start` is at most `WIN`
/// (the caller passes `ingest_lo .. anchor_pos`, which is).
pub fn win_slices(start: i64, end: i64) -> Vec<WinSlice> {
    let mut out = Vec::new();
    if end <= start {
        return out;
    }
    let mut p = start;
    while p < end {
        let phys = (p as usize) % WIN;
        let room = WIN - phys;
        let n = ((end - p) as usize).min(room);
        out.push(WinSlice { phys, pos0: p, n });
        p += n as i64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_is_the_anchor_then_seven_masks() {
        let ids = block_ids(1001, MASK);
        assert_eq!(ids[0], 1001);
        assert!(ids[1..].iter().all(|&x| x == MASK));
        assert_eq!(ids.len(), 8);
    }

    #[test]
    fn rope_and_attention_name_the_same_position() {
        let anchor_pos = 27_290i64;
        let pos = block_positions(anchor_pos);
        for t in 0..BLOCK {
            assert_eq!(pos[t], attention_position(anchor_pos, t));
        }
        // the off-by-one the previous port shipped: starts = anchor_pos + 1, bpos = anchor_pos + t
        assert_ne!(attention_position(anchor_pos + 1, 0), pos[0]);
    }

    #[test]
    fn the_anchor_is_not_context() {
        assert_eq!(context_from(100), 0);
        assert_eq!(ingest_lo(0, 100), 0);
        assert_eq!(context_from(3_000), 3_000 - WIN as i64);
        // a fresh ring (until 0) on a long prompt ingests the window, not positions 0..anchor
        assert_eq!(ingest_lo(0, 3_000), 952);
        // the next round ingests only the tokens the last verify kept
        assert_eq!(ingest_lo(3_000, 3_008), 3_000);
        // a gap wider than the window restarts at the window, the overwritten slots are not read
        assert_eq!(ingest_lo(100, 3_000), 952);
        assert_eq!(ingest_lo(3_008, 3_008), 3_008);
    }

    #[test]
    fn a_prefill_or_a_rewind_drops_the_ring() {
        // a full accept advances by one block and keeps the keys
        assert_eq!(resume_until(3_000, 3_008), 3_000);
        assert_eq!(resume_until(3_000, 3_001), 3_000);
        // one past a block is a new span (the next prompt), not eight accepted drafts
        assert_eq!(resume_until(3_000, 3_009), 0);
        assert_eq!(resume_until(4_900, 8_000), 0);
        assert_eq!(resume_until(4_900, 100), 0);
        assert_eq!(resume_until(0, 8_000), 0);
    }

    #[test]
    fn a_window_that_wraps_the_capture_is_two_slices() {
        assert_eq!(win_slices(0, 100), vec![WinSlice { phys: 0, pos0: 0, n: 100 }]);
        assert_eq!(win_slices(0, WIN as i64), vec![WinSlice { phys: 0, pos0: 0, n: WIN }]);
        assert_eq!(win_slices(2_048, 2_058), vec![WinSlice { phys: 0, pos0: 2_048, n: 10 }]);
        let s = win_slices(952, 3_000);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], WinSlice { phys: 952, pos0: 952, n: WIN - 952 });
        assert_eq!(s[1], WinSlice { phys: 0, pos0: 2_048, n: 952 });
        assert_eq!(s[0].n + s[1].n, WIN);
        assert!(win_slices(5, 5).is_empty());
    }
}

#!/usr/bin/env python3
"""The host half of the DFlash2 block contract.

Same cases as tidelm-dflash's plan_tests. They do not need a GPU: the block is the anchor
then seven mask tokens, the context fold is the window and then only the new tail, and the
next anchor sits one past the drafts that were kept. The kernels are not run here.
"""
MASK = 154856
BLOCK = 8
WIN = 2048


def block_plan(anchor, anchor_pos):
    ids = [MASK] * BLOCK
    ids[0] = anchor
    pos = [anchor_pos + t for t in range(BLOCK)]
    start = anchor_pos
    frm = max(0, anchor_pos - (WIN - 1))
    return ids, pos, start, frm


def ingest_span(seeded_to, anchor_pos, trust_lo=0):
    if anchor_pos <= 0:
        return None
    hi = anchor_pos
    window_lo = max(0, hi - WIN)
    if seeded_to < 0 or seeded_to > hi:
        lo = window_lo
    else:
        lo = max(seeded_to, window_lo)
    if trust_lo > lo:
        lo = trust_lo
    if lo >= hi:
        return None
    return lo, hi


def next_anchor_pos(anchor_pos, n_acc):
    return anchor_pos + 1 + n_acc


def test_block():
    ids, pos, start, frm = block_plan(1001, 50)
    assert ids[0] == 1001
    assert ids[1:] == [MASK] * 7
    assert pos == [50, 51, 52, 53, 54, 55, 56, 57]
    assert start == 50
    assert frm == 0
    _, _, far_start, far_from = block_plan(7, 3000)
    assert far_from == 3000 - 2047
    assert far_start == 3000


def test_context():
    assert ingest_span(-1, 100) == (0, 100)
    assert ingest_span(-1, 5000) == (5000 - 2048, 5000)
    assert ingest_span(100, 108) == (100, 108)
    assert ingest_span(108, 108) is None
    assert ingest_span(0, 0) is None
    assert ingest_span(5000, 100) == (0, 100)
    assert ingest_span(100, 3000) == (3000 - 2048, 3000)
    # a rewind kept the prefix; only positions this request retapped are folded
    assert ingest_span(-1, 10020, 9998) == (9998, 10020)


def test_next_anchor():
    assert next_anchor_pos(100, 0) == 101
    assert next_anchor_pos(100, 3) == 104
    assert ingest_span(100, next_anchor_pos(100, 3)) == (100, 104)


if __name__ == "__main__":
    test_block()
    test_context()
    test_next_anchor()
    print("dflash plan: 3 checks ok")

#!/usr/bin/env python3
"""concurrency_probe.py -- does serving several unrelated conversations at once cost speed?

Four threads, four topics, four different preambles (so no thread reuses another's prefix and the
prefix cache cannot answer for them). Each request is streamed: TTFT is its prefill, the rest is its
decode, and the run also reports the wall-clock throughput of all four together against the same
requests run one at a time.

Usage: python3 tools/concurrency_probe.py [--base http://127.0.0.1:8083] [--model GLM-5.3-Flash]
                                          [--threads 4] [--isl 2000] [--osl 256] [--repeat 2]
Exit 0 if every request finished; the numbers are printed either way.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

TOPICS = [
    ("harbour", """You are a marine engineer writing for a harbour authority. The following is the
survey report of a nineteenth-century stone pier, its tide tables and the silting of its basin.
""", """Explain, for the harbour board, what the survey implies for next winter's dredging budget: which
sections must be repaired first, what the tide tables show about the basin's mouth, and how long the
works can run before the spring tides close the window."""),
    ("compiler", """You are a compiler engineer documenting an SSA-based optimiser for a small imperative
language. The following notes describe its passes, the dominance frontier computation and the register
allocator's interference graph.
""", """Write the section of the design document that explains how the dominating passes interact: what
the dead-code pass may remove, what the loop pass may then hoist, and where the allocator has to give
up on coalescing."""),
    ("bread", """You are a baker writing a technical sheet about sourdough fermentation: cultures, hydration,
dough temperature, folding schedules and the chemistry of the crumb.
""", """Write the sheet's troubleshooting half: a dense, flat crumb, a pale crust, a too-sour loaf and a
loaf that tears at the base -- the cause of each and the change to the schedule that fixes it."""),
    ("treaty", """You are a legal historian annotating a treaty on river navigation signed in the eighteen
hundreds: its articles on tolls, pilots, winter closure and the arbitration of disputes.
""", """Annotate the articles on arbitration: what the drafters meant by a dispute of navigation, how the
commission was to be composed, and why the winter-closure clause was the one that failed first."""),
]

FILLER = ("The record continues in the same hand, page after page, noting the weather, the state of the "
          "works, the men employed and the sums spent, and it is only by reading the whole of it that "
          "the pattern becomes plain. ")


def build(topic, isl_chars):
    head, ask = topic[1], topic[2]
    body = head + (FILLER * (isl_chars // len(FILLER) + 1))[:isl_chars]
    return body + "\n\n" + ask


def run(base, model, prompt, osl, out):
    """One streamed request: out gets ttft_s, decode_tok_s, out_tokens, prompt_tokens, total_s."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": osl, "temperature": 0.4, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    tokens = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
    total = time.time() - t0
    n = usage.get("completion_tokens", tokens) or tokens
    out.update(ttft_s=ttft, total_s=total, out_tokens=n,
               prompt_tokens=usage.get("prompt_tokens"),
               decode_tok_s=(n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8083")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--isl", type=int, default=2000, help="prompt characters per thread")
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    model = a.model or json.load(urllib.request.urlopen(
        urllib.request.Request(a.base.rstrip("/") + "/v1/models"), timeout=30))["data"][0]["id"]
    topics = TOPICS[: a.threads % len(TOPICS) or len(TOPICS)] if a.threads <= len(TOPICS) else TOPICS
    prompts = [build(topics[i % len(topics)], a.isl) for i in range(a.threads)]

    def one(i):
        d = {}
        try:
            run(a.base, model, prompts[i], a.osl, d)
        except Exception as e:  # noqa: BLE001 -- reported, not raised: a failed thread is a result
            d["error"] = repr(e)[:200]
        return d

    def batch():
        res = [None] * len(prompts)
        def worker(i):
            res[i] = one(i)
        t0 = time.time()
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return res, time.time() - t0

    print(f"model {model}: {len(prompts)} prompts of ~{a.isl} chars, {a.osl} tokens each")
    serial, serial_wall = [], 0.0
    for _ in range(a.repeat):
        for i in range(len(prompts)):
            t0 = time.time()
            serial.append(one(i))
            serial_wall += time.time() - t0
    conc, conc_wall = [], 0.0
    for _ in range(a.repeat):
        r, w = batch()
        conc += r
        conc_wall += w

    def med(rows, key):
        v = [r[key] for r in rows if r and r.get(key)]
        return statistics.median(v) if v else float("nan")

    print("\n  mode     ttft_med_s  decode_med_tok/s  out_tokens  wall_s  tok/s(all)")
    for name, rows, wall in (("serial", serial, serial_wall), ("parallel", conc, conc_wall)):
        tok = sum(r.get("out_tokens") or 0 for r in rows if r)
        print(f"  {name:<9} {med(rows, 'ttft_s'):>9.2f}  {med(rows, 'decode_tok_s'):>16.2f}  "
              f"{tok:>10}  {wall:>6.1f}  {tok / wall if wall else float('nan'):>8.2f}")
    print()
    for i, r in enumerate(conc):
        if r and r.get("error"):
            print(f"  thread {i}: ERROR {r['error']}")
        elif r:
            print(f"  thread {i}: prompt {r.get('prompt_tokens')} ttft {r.get('ttft_s', 0):.2f} s "
                  f"decode {r.get('decode_tok_s') or float('nan'):.2f} tok/s out {r.get('out_tokens')}")
    bad = [r for r in conc if not r or r.get("error")]
    print(f"\n  threads ok {len(conc) - len(bad)}/{len(conc)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())

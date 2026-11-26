#!/usr/bin/env python3
"""The EXL3 prefill oracle: one expert's gate/up pair and down projection, as the kernels do it.

Numerics mirror the fused prompt path (TensorFold's ``exl3_prompt.cuh``, fp16 MMA, and its
e4m3 variant ``exl3_prompt_x8.cuh``) in numpy, so a CUDA test compares against this file
instead of reimplementing the format.  The decode is ``tools/exl3_format.py`` and the pack
reader ``tools/exl3_ref.py`` (imported, not copied): the oracle adds only the kernels' order
of operations.

One pair (one slot row p, one expert e), act_mode 0 (the served mode), as the kernels do it:

  rotate-in   v[j] = fp32(x[p,k]) * fp32(suh[k])   (x read as bf16, suh as fp16)
              fwht128(v)                           (fp32, the 128-butterfly order)
              X0 = fp16(v * HAD_SCALE)             (one rounding at the output)
  gate/up     acc = X0 @ W_q^T                     (fp16 operands, fp32 accumulate: m16n8k16)
              gv = fwht128(acc_g) * HAD_SCALE * fp32(svh_g[n])
              gg = bf16r(gv);  uu = bf16r(fwht128(acc_u) * HAD_SCALE * fp32(svh_u[n]))
              if limit: gg = upTo(gg, limit); uu = within(uu, limit)
              act = bf16r( bf16r(gg / (1 + exp(-gg))) * uu )
              v = act * fp32(suh_d[n])
              Xd = fp16(fwht128(v) * HAD_SCALE)    (one rounding)
  down        accd = Xd @ Wd_q^T                   (fp16 operands, fp32 accumulate)
              y[p] += fwht128(accd) * HAD_SCALE * fp32(svh_d[k])   (fp32 y, one expert's share)

The e4m3 variant (``--variant x8``, TF_GLM_X3_FP8) changes exactly two steps.  Rotate-in keeps
the fp16 rounding and reads it back: ``v = fp32(fp16(fwht128(...) * HAD_SCALE))``, the row's
scale is its absmax / 448 (0 rows keep scale 1), and the staged byte is e4m3-satfinite of
``v / scale``.  The weights: each decoded fp16 goes through ``cvt.rn.satfinite.e4m3x2.f16x2``.
The MMA is m16n8k32 over e4m3 pairs, so the accumulator is fp32 over e4m3 products; the
epilogue multiplies the accumulators by the row scales first (``gv = acc_g * gs``, fp32, not
bf16-rounded before the fwht), then runs the same act_mode-0 epilogue.  The down projection
is untouched (fp16 Xd @ fp16 Wd).  The kernel permutes K within each 32-chunk into the A
fragment's logical order on both X and W; a consistent permutation of both sides does not
change the sum's value, only its pairwise order, so the oracle computes the plain sum.

The fp32 reference (``--variant fp32``) runs the same steps in float64 with no roundings: the
number every tolerance is stated against.

CLI: ``--pack PATH --layer L --expert E --rows R [--seed S] [--limit F] [--variant fp16|x8|fp32]
[--emit npz --out FILE] [--compare FILE]``.  The emit's npz carries the intermediate arrays
(x rows, the rotated rows, the epilogue's gate/up accumulator, Xd, y) so a CUDA test diffuses
step by step instead of only at the end.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import exl3_format as F                       # noqa: E402  (the vendored decode oracle)
import exl3_ref as R                          # noqa: E402  (the pack reader)

HAD_SCALE = float(np.float32(0.08838834764831845))     # exl3_decode.cuh's constant
E4M3_MAX = 448.0                                       # e4m3's largest finite magnitude
VARIANTS = ("fp16", "x8", "fp32")


# --- the roundings (the kernels' intrinsics, numpy) --------------------------------------------
def bf16r(x: np.ndarray) -> np.ndarray:
    """__float2bfloat16_rn: round-to-nearest-even to bf16, read back as fp32."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # truncation, then the RN bit: biased up when the dropped 16 bits are past the midpoint
    r = ((u >> 16) & 1) + 0x7FFF
    out = ((u + r) & 0xFFFF0000).view(np.float32)
    return out


def fp16r(x: np.ndarray) -> np.ndarray:
    """__float2half_rn."""
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def fp16_f32(x: np.ndarray) -> np.ndarray:
    """fp16 store, then fp32 read back (__float2half_rn followed by __half2float)."""
    return fp16r(x).astype(np.float32)


def e4m3(x: np.ndarray) -> np.ndarray:
    """__nv_cvt_float_to_fp8(.., SATFINITE, E4M3) as an exact float: the nearest representable
    e4m3 value, ties to even mantissa, magnitudes past 448 clamped (NaN stays NaN)."""
    x = np.asarray(x, dtype=np.float32)
    nan = np.isnan(x)
    mag = np.clip(np.abs(np.where(nan, 0.0, x)), 0.0, E4M3_MAX)
    code = _e4m3_code(mag)
    return np.where(nan, np.float32(np.nan), code * np.sign(np.where(nan, 1.0, x))).astype(np.float32)


def _e4m3_code(mag: np.ndarray) -> np.ndarray:
    """The magnitude of the nearest e4m3 value (the 128 positive codes, ties to even)."""
    # e4m3: exponent bias 7, 3 mantissa bits; subnormals step 2^-9, normals 2^(e-10).
    m = mag.astype(np.float64)
    m = np.maximum(m, 2.0 ** -9)
    exp = np.floor(np.log2(m))
    normal = exp >= -6
    # subnormal: quantise on the 2^-9 grid directly (round-half-even)
    sub = np.round(m / 2.0 ** -9) * 2.0 ** -9
    # normal: mantissa on the 2^(e-10) grid; np.round is round-half-even
    e = np.maximum(exp, -6)
    q = np.round(m / 2.0 ** (e - 10))
    norm = q * 2.0 ** (e - 10)
    out = np.where(normal, norm, sub)
    # a mantissa that rounded up to 16.0 carries into the next binade: recompute there
    carry = q > 15.0
    if np.any(carry):
        m2 = np.maximum(np.where(carry, norm, m), 2.0 ** -9)
        exp2 = np.floor(np.log2(m2))
        e2 = np.maximum(exp2, -6)
        q2 = np.round(m2 / 2.0 ** (e2 - 10))
        out = np.where(carry, np.minimum(q2 * 2.0 ** (e2 - 10), E4M3_MAX), out)
    return np.minimum(out, E4M3_MAX).astype(np.float32)


# --- the Hadamard rotation (exl3_decode.cuh's fwht128 + HAD_SCALE) ------------------------------
@F.lru_cache(maxsize=1)
def _fwht_log() -> np.ndarray:
    """The 128x128 float32 matrix fwht128 applies: the Sylvester Hadamard, the butterfly's exact sum."""
    return F.hadamard().astype(np.float32)


def fwht128(x: np.ndarray) -> np.ndarray:
    """H over every block of 128 along the last axis, fp32, the kernel's butterfly order."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % 128:
        raise ValueError(f"the rotated dimension must be a multiple of 128, got {n}")
    y = x.reshape(x.shape[:-1] + (n // 128, 128))
    return (y @ _fwht_log()).reshape(x.shape)


def rotate_in(x_row: np.ndarray, suh: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """tf_exl3_experts_rot_in / tf_exl3_prompt_rotate / tf_exl3_experts_rot_in8.

    x_row [K] (the caller has applied bf16 storage), suh fp16 [K].  Returns (X0 [K] fp32 as
    the MMA reads it, scale) - the scale is 1.0 unless variant == 'x8'."""
    v = x_row.astype(np.float32) * suh.astype(np.float32)
    v = fwht128(v)
    x0 = fp16_f32(v * np.float32(HAD_SCALE))                   # the fp16 rounding, read back
    if variant != "x8":
        return x0, np.float32(1.0)
    mx = float(np.abs(x0).max())
    sc = np.float32(mx / 448.0) if mx > 0.0 else np.float32(1.0)
    x8 = e4m3(x0 / sc)
    return x8, sc


def mma(a_f16: np.ndarray, w_f16: np.ndarray, variant: str) -> np.ndarray:
    """X [rows, K] (as the MMA fragments read it) @ W_q^T [K, N] -> [rows, N] fp32 accumulate.

    fp16: the operands are fp16, the products fp32 (m16n8k16).  x8: the operands are e4m3
    (stored fp16 here), the products fp32 (m16n8k32).  The fp32 sum over K is the MMA's
    accumulator; the oracle's np.matmul sums in its own order, which is the tile-order
    tolerance Deliverable 2 measures."""
    a = a_f16.astype(np.float32)
    b = w_f16.astype(np.float32)
    return a @ b


def decode_w(rec: dict, bits: list[float], codebook: str) -> dict[str, np.ndarray]:
    """The record's trellises decoded to fp16 W_q [K, N] (exl3_format.unpack, bit-exact)."""
    w1 = F.unpack(rec["t1"], bits[0], codebook)                # gate: [K, N]
    w3 = F.unpack(rec["t3"], bits[1], codebook)                # up:   [K, N]
    w2 = F.unpack(rec["t2"], bits[2], codebook)                # down: [N2=K, K2=N] -> [K, N] of the down pair
    return {"w1": w1, "w3": w3, "w2": w2}


def gateup_epilogue(acc_g: np.ndarray, acc_u: np.ndarray, svh_g: np.ndarray, svh_u: np.ndarray,
                    suh_d: np.ndarray, limit: float, act_mode: int) -> np.ndarray:
    """tf_exl3_prompt_gu's epilogue: rotate out, * svh, clamp, SwiGLU, * suh_d, rotate, fp16.

    acc_g / acc_u [rows, N] fp32 (x8: already multiplied by the row scales by the caller);
    svh_g / svh_u / suh_d fp16 [N] (one expert's rows).  Returns Xd [rows, N] fp32 as fp16."""
    gv = fwht128(acc_g) * np.float32(HAD_SCALE) * svh_g.astype(np.float32)[None, :]
    uu_in = fwht128(acc_u) * np.float32(HAD_SCALE) * svh_u.astype(np.float32)[None, :]
    if act_mode == 0:
        gg = bf16r(gv)
        uu = bf16r(uu_in)
    else:
        gg, uu = gv, uu_in
    if limit > 0:
        gg = np.minimum(gg, limit)                             # upTo: only the upper side
        uu = np.clip(uu, -limit, limit)                        # within: both sides
    act = gg / (1.0 + np.exp(-gg))
    if act_mode == 0:
        act = bf16r(act)
        act = bf16r(act * uu)
    else:
        act = act * uu
    v = act * suh_d.astype(np.float32)[None, :]
    return fp16_f32(fwht128(v) * np.float32(HAD_SCALE))


def down_combine(acc_d: np.ndarray, svh_d: np.ndarray, y: np.ndarray) -> np.ndarray:
    """tf_exl3_prompt_down's epilogue: rotate out, * svh_d, add into y [rows, K2] fp32."""
    v = fwht128(acc_d) * np.float32(HAD_SCALE) * svh_d.astype(np.float32)[None, :]
    return y + v


# --- the pair's forward ------------------------------------------------------------------------
def expert_pair(pack: R.PackReader, layer: int, expert: int, x_rows: np.ndarray, *,
                limit: float = 10.0, act_mode: int = 0, variant: str = "fp16",
                trace: bool = False) -> dict[str, np.ndarray]:
    """One expert's pair over ``rows`` input rows.  x_rows [rows, K] already bf16-stored (fp32).

    Returns {'y': [rows, K] fp32 (the expert's share), 'xd', 'x_rot_g', 'x_rot_u',
    'acc_g', 'acc_u', 'acc_d', 'row_scale_g', 'row_scale_u'}; the caller combines the
    experts' shares by their routing weights (fp32 fmaf, the server's combine)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    R.set_dims(pack.header["dims"][0], pack.header["dims"][1])
    rec = pack.read_expert(layer, expert)
    bits = pack.records[(layer, expert)]["bits"]
    if not isinstance(bits, (list, tuple)):
        bits = [bits, bits, bits]
    bits = [float(b) for b in bits]
    cb = pack.codebook
    K, N = pack.header["dims"][0], pack.header["dims"][1]

    w = decode_w(rec, bits, cb)
    if variant == "x8":
        # cvt.rn.satfinite.e4m3x2.f16x2 on the decoded fp16 weights
        w["w1"] = fp16_f32(e4m3(w["w1"].astype(np.float32))).astype(np.float16)
        w["w3"] = fp16_f32(e4m3(w["w3"].astype(np.float32))).astype(np.float16)
    w1 = w["w1"].astype(np.float32)
    w3 = w["w3"].astype(np.float32)
    w2 = w["w2"].astype(np.float32)

    suh1, suh3 = rec["suh1"], rec["suh3"]
    suh2, svh1, svh3, svh2 = rec["suh2"], rec["svh1"], rec["svh3"], rec["svh2"]

    out = {"row_scale_g": np.ones(x_rows.shape[0], dtype=np.float32),
           "row_scale_u": np.ones(x_rows.shape[0], dtype=np.float32)}
    if variant == "fp32":
        # HAD_SCALE enters twice per projection like the fp16 path (rotate-in AND epilogue):
        # y = x @ (diag(suh) @ (H/sqrt(128)) @ W_q @ (H/sqrt(128)) @ diag(svh)), not once.
        scale = 1.0 / math.sqrt(128.0)
        xg = fwht128(x_rows * suh1.astype(np.float64)[None, :]) * scale
        xu = fwht128(x_rows * suh3.astype(np.float64)[None, :]) * scale
        acc_g = xg @ w1.astype(np.float64)
        acc_u = xu @ w3.astype(np.float64)
        gg = fwht128(acc_g) * scale * svh1.astype(np.float64)[None, :]
        uu = fwht128(acc_u) * scale * svh3.astype(np.float64)[None, :]
        if limit > 0:
            gg = np.minimum(gg, limit)
            uu = np.clip(uu, -limit, limit)
        act = gg / (1.0 + np.exp(-gg)) * uu
        v = act * suh2.astype(np.float64)[None, :]
        xd = fwht128(v) * scale
        acc_d = xd @ w2.astype(np.float64)
        y = fwht128(acc_d) * scale * svh2.astype(np.float64)[None, :]
        out.update(y=y.astype(np.float32), xd=xd.astype(np.float32),
                   x_rot_g=xg.astype(np.float32), x_rot_u=xu.astype(np.float32),
                   acc_g=acc_g.astype(np.float32), acc_u=acc_u.astype(np.float32),
                   acc_d=acc_d.astype(np.float32))
        return out

    # rotate-in, the two matrices' rows (one scale each when x8)
    xg_list, xu_list, sg_list, su_list = [], [], [], []
    for r in range(x_rows.shape[0]):
        a, sa = rotate_in(x_rows[r], suh1, variant)
        b, sb = rotate_in(x_rows[r], suh3, variant)
        xg_list.append(a); xu_list.append(b); sg_list.append(sa); su_list.append(sb)
    xg = np.stack(xg_list)
    xu = np.stack(xu_list)
    sg = np.stack(sg_list)
    su = np.stack(su_list)
    out["row_scale_g"], out["row_scale_u"] = sg, su
    if trace:
        out["x_rot_g"], out["x_rot_u"] = xg, xu

    acc_g = mma(xg, w1, variant)
    acc_u = mma(xu, w3, variant)
    if variant == "x8":
        acc_g = acc_g * sg[:, None]
        acc_u = acc_u * su[:, None]
    if trace:
        out["acc_g"], out["acc_u"] = acc_g, acc_u

    xd = gateup_epilogue(acc_g, acc_u, svh1, svh3, suh2, limit, act_mode)
    if trace:
        out["xd"] = xd

    acc_d = mma(xd, w2, variant)                               # fp16 down weights either way
    if trace:
        out["acc_d"] = acc_d
    y = down_combine(acc_d, svh2, np.zeros((x_rows.shape[0], K), dtype=np.float32))
    out["y"] = y
    return out


# --- the CLI -----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="The EXL3 prefill oracle: one expert's gate/up pair and"
                                             " down projection as the fused kernels compute them")
    ap.add_argument("--pack", required=True, help="path to the EXL3 experts pack (exl3-experts format)")
    ap.add_argument("--layer", type=int, required=True, help="the layer (the pack's record key L)")
    ap.add_argument("--expert", type=int, required=True, help="the expert id (the pack's record key ,E)")
    ap.add_argument("--rows", type=int, default=16, choices=range(16, 65),
                    help="input rows, 16 to 64 (default 16)")
    ap.add_argument("--seed", type=int, default=0, help="the synthetic rows' seed (default 0)")
    ap.add_argument("--limit", type=float, default=10.0, help="the SwiGLU clamp (default 10.0, 0: none)")
    ap.add_argument("--act-mode", type=int, default=0, choices=(0, 1),
                    help="0: bf16 roundings (the served mode), 1: fp32")
    ap.add_argument("--variant", choices=VARIANTS, default="fp16",
                    help="fp16: the fp16-MMA kernels; x8: their e4m3 variant; fp32: the float64 reference")
    ap.add_argument("--emit", choices=("npz",), default=None,
                    help="write the intermediates (x, y, xd, x_rot_g/u, acc_g/u/d, row scales) as npz to --out")
    ap.add_argument("--out", default=None, help="the emit path (required with --emit)")
    ap.add_argument("--compare", default=None,
                    help="a previous emit's npz: print the max/rms delta of y against it")
    args = ap.parse_args()
    if args.emit and not args.out:
        ap.error("--emit requires --out")

    pack = R.PackReader(args.pack)
    rng = np.random.default_rng(args.seed)
    K = pack.header["dims"][0]
    x = (rng.standard_normal((args.rows, K)) * 0.5)
    x_rows = bf16r(x.astype(np.float32))                       # the model's rows are bf16-stored
    res = expert_pair(pack, args.layer, args.expert, x_rows, limit=args.limit,
                      act_mode=args.act_mode, variant=args.variant, trace=True)

    report = {"pack": args.pack, "layer": args.layer, "expert": args.expert, "rows": args.rows,
              "seed": args.seed, "limit": args.limit, "act_mode": args.act_mode,
              "variant": args.variant,
              "bits": [float(b) for b in pack.records[(args.layer, args.expert)]["bits"]],
              "y_absmax": float(np.abs(res["y"]).max()),
              "y_rms": float(np.sqrt((res["y"].astype(np.float64) ** 2).mean()))}
    if args.compare:
        ref = np.load(args.compare)
        d = res["y"].astype(np.float64) - ref["y"].astype(np.float64)
        report["compare"] = args.compare
        report["max_abs_delta"] = float(np.abs(d).max())
        report["rms_delta"] = float(np.sqrt((d ** 2).mean()))
        ref_rms = float(np.sqrt((ref["y"].astype(np.float64) ** 2).mean()))
        report["rel_rms"] = report["rms_delta"] / max(ref_rms, 1e-30)
    print(json.dumps(report, indent=2))
    if args.emit:
        np.savez(args.out, x=x_rows, y=res["y"], xd=res["xd"],
                 x_rot_g=res["x_rot_g"], x_rot_u=res["x_rot_u"],
                 acc_g=res["acc_g"], acc_u=res["acc_u"], acc_d=res["acc_d"],
                 row_scale_g=res["row_scale_g"], row_scale_u=res["row_scale_u"],
                 meta=json.dumps(report))
        print(f"emitted {args.out}")


if __name__ == "__main__":
    main()

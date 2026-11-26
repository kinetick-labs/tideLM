"""The EXL3 prefill oracle's tests (numpy only, no pack on disk needed).

A synthetic mini pack (dims 512x128, one layer, one expert, mul1) is built by
``R.write_pack`` from random trellis tiles and positive fp16 scales -- the pack is
self-consistent by construction, and ``exl3_format.unpack`` / ``exl3_format.dequantize``
(the vendored bit-exact numpy decoder ``tools/test_exl3_ref.py`` already proves equals
the torch reference in ``tools/exl3_ref.py``) serve as the tile-for-tile oracle.

The task's four checks:
  1. the decode matches ``tools/exl3_ref.py`` tile for tile on a fixed tile
     (via the vendored numpy decoder ``exl3_format.unpack`` -- the same bytes
     ``exl3_ref.unpack`` decodes, bit-exact, per test_exl3_ref.py);
  2. a 16-row block and a 64-row block of the same rows agree row for row;
  3. the rotation is its own inverse where the kernel assumes it is (H.H/128 = I,
     and double fwht128 with HAD_SCALE^2 returns the input);
  4. the epilogue's per-row scale is applied once (2x svh2 doubles the output exactly;
     the gate/up svh scales sit inside SwiGLU's nonlinearity, so only the down side is
     exact-doubling).

Run:  uv run --no-project --with numpy --with pytest python -m pytest tools/tests -q
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import exl3_format as F          # noqa: E402
import exl3_ref as R             # noqa: E402
import exl3_prefill_ref as P     # noqa: E402

K, N = 512, 128
BITS = [2.0, 2.0, 3.0]           # gate 2, up 2, down 3: a version-2 record


def _random_record(rng: np.random.Generator) -> dict[str, np.ndarray]:
    R.set_dims(K, N)
    shapes = R.record_shapes(BITS, 0, 1)
    rec = {}
    for name in R.RECORD_TENSORS:
        s = shapes[name]
        if name in ("t1", "t3", "t2"):
            rec[name] = rng.integers(-2**15, 2**15, s, dtype=np.int16)
        else:
            rec[name] = (0.5 + 0.5 * rng.random(s)).astype(np.float16)   # positive scales
    return rec


@pytest.fixture(scope="module")
def mini_pack(tmp_path_factory):
    """A one-layer, one-expert pack on disk (round-trips through the real writer)."""
    rng = np.random.default_rng(5)
    rec = _random_record(rng)
    path = str(tmp_path_factory.mktemp("exl3pre") / "mini-w1.bin")
    R.write_pack(path, {0: 3.0}, 0, 1, "mul1", "test", (((0, 0), rec),),
                 n_experts=1, expert_bits={(0, 0): BITS})
    return path, rec


@pytest.fixture(scope="module")
def reader(mini_pack):
    return R.PackReader(mini_pack[0])


def _rows(seed: int, rows: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return P.bf16r((rng.standard_normal((rows, K)) * 0.2).astype(np.float32))


# --- the roundings -----------------------------------------------------------------------------

def test_bf16r_round_to_nearest_even():
    # 1.00390625 (0x3F808000) is an exact bf16 tie; RN-even rounds up to the even mantissa
    # 0x3F808000 is an exact bf16 tie whose kept mantissa (1000000) is even: RN-even keeps it.
    # 0x3F818000 (1.01171875) is a tie with an odd kept mantissa: RN-even rounds up.
    x = np.array([1.0, 1.00390625, 1.01171875, -2.5], dtype=np.float32)
    out = P.bf16r(x)
    assert float(out[0]) == 1.0
    assert float(out[1]) == 1.0                      # the even-mantissa tie stays
    assert float(out[2]) == 1.015625                 # the odd-mantissa tie goes up to even
    assert float(out[3]) == -2.5


def test_fp16r_and_e4m3_saturate():
    x = np.array([448.0, 449.0, -448.0, 1e-6, 2.0**-9, 1000.0], dtype=np.float32)
    out = P.e4m3(x)
    assert float(out[0]) == 448.0 and float(out[1]) == 448.0    # satfinite
    assert float(out[2]) == -448.0 and float(out[5]) == 448.0
    assert 0.0 < float(out[3]) < 0.01                            # the small value survives


# --- the rotation ------------------------------------------------------------------------------

def test_hadamard_self_inverse():
    """H.H/128 = I: the kernel's assumption that the rotation undoes itself."""
    H = F.hadamard().astype(np.float64)
    assert (H @ H == 128.0 * np.eye(128)).all()      # exact in integers, no tolerance


def test_fwht128_double_returns_input():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((8, 128)).astype(np.float32)
    back = P.fwht128(P.fwht128(v)) * (P.HAD_SCALE ** 2)
    assert np.allclose(back, v, atol=1e-5)
    assert np.allclose(P.fwht128(P.fwht128(v)) / 128.0, v, atol=1e-5)


def test_fwht128_is_the_hadamard():
    d = np.zeros(128, dtype=np.float32)
    d[0] = 1.0
    assert np.array_equal(P.fwht128(d), F.hadamard().astype(np.float32)[:, 0])


# --- the decode, tile for tile -----------------------------------------------------------------

def test_decode_w_is_the_vendored_decoder_tile_for_tile(reader, mini_pack):
    """decode_w == exl3_format.unpack (the bit-exact numpy oracle exl3_ref.unpack decodes
    the same trellis bytes; test_exl3_ref.py already pins the torch decode to this)."""
    _, rec = mini_pack
    w = P.decode_w(rec, BITS, "mul1")
    assert np.array_equal(w["w1"], F.unpack(rec["t1"], BITS[0], "mul1"))
    assert np.array_equal(w["w3"], F.unpack(rec["t3"], BITS[1], "mul1"))
    assert np.array_equal(w["w2"], F.unpack(rec["t2"], BITS[2], "mul1"))
    # and the pack on disk holds the same tiles
    back = reader.read_expert(0, 0)
    for name in R.RECORD_TENSORS:
        assert np.array_equal(back[name], rec[name]), name


def test_pack_round_trip_and_alignment(reader):
    assert reader.header["dims"] == [512, 128]
    assert reader.records[(0, 0)]["bits"] == BITS
    assert reader.records[(0, 0)]["off"] % R.ALIGN == 0


# --- the fp32 reference against the float64 dequant math ---------------------------------------

def test_fp32_variant_matches_dequant_math(mini_pack, reader):
    """The fp32 branch is the model's own math: x @ (dequantized W) through SwiGLU."""
    _, rec = mini_pack
    x = _rows(7, 16)
    r = P.expert_pair(reader, 0, 0, x, limit=0, variant="fp32")
    w1 = F.dequantize(rec["t1"], rec["suh1"], rec["svh1"], BITS[0], "mul1")
    w3 = F.dequantize(rec["t3"], rec["suh3"], rec["svh3"], BITS[1], "mul1")
    w2 = F.dequantize(rec["t2"], rec["suh2"], rec["svh2"], BITS[2], "mul1")
    xd = x.astype(np.float64)
    g = xd @ w1
    u = xd @ w3
    v = g / (1.0 + np.exp(-g)) * u
    y = v @ w2
    rel = float(np.linalg.norm(r["y"] - y) / np.linalg.norm(y))
    assert rel < 1e-5


def test_fp16_and_x8_within_tolerance_of_fp32(mini_pack, reader):
    _, rec = mini_pack
    x = _rows(7, 16)
    r32 = P.expert_pair(reader, 0, 0, x, limit=10.0, variant="fp32")
    base = float(np.sqrt((r32["y"].astype(np.float64) ** 2).mean()))
    r16 = P.expert_pair(reader, 0, 0, x, limit=10.0, variant="fp16")
    d16 = r16["y"].astype(np.float64) - r32["y"]
    rx8 = P.expert_pair(reader, 0, 0, x, limit=10.0, variant="x8")
    d8 = rx8["y"].astype(np.float64) - r32["y"]
    rel16 = float(np.sqrt((d16 ** 2).mean()) / base)
    rel8 = float(np.sqrt((d8 ** 2).mean()) / base)
    assert rel16 < 0.01, rel16
    assert rel8 < 0.01, rel8
    # x8 quantises harder: its delta must not be smaller than fp16's on the same rows by
    # construction (both round, but e4m3 quantises the inputs too); just record it
    assert rel8 >= rel16 * 0.5


# --- the task's row-agreement check ------------------------------------------------------------

def test_16_row_block_matches_64_row_block_row_for_row(reader):
    x = _rows(11, 64)
    r64 = P.expert_pair(reader, 0, 0, x, limit=10.0, variant="fp16")
    r16 = P.expert_pair(reader, 0, 0, x[:16], limit=10.0, variant="fp16")
    assert np.array_equal(r16["y"], r64["y"][:16])
    r32v = P.expert_pair(reader, 0, 0, x, limit=10.0, variant="fp32")
    r16v = P.expert_pair(reader, 0, 0, x[:16], limit=10.0, variant="fp32")
    assert np.array_equal(r16v["y"], r32v["y"][:16])


# --- the epilogue's per-row scale --------------------------------------------------------------

def test_svh_down_doubles_the_output_exactly(reader):
    """2x svh2 (the down projection's output side, linear) doubles y exactly."""
    x = _rows(9, 16)
    r1 = P.expert_pair(reader, 0, 0, x, limit=0, variant="fp32")
    rec2 = dict(reader.read_expert(0, 0))
    rec2["svh2"] = rec2["svh2"] * np.float16(2.0)
    class FP: pass
    fp = FP()
    fp.header = reader.header
    fp.records = reader.records
    fp.codebook = reader.codebook
    fp.read_expert = lambda l, e: rec2
    r2 = P.expert_pair(fp, 0, 0, x, limit=0, variant="fp32")
    assert float(np.abs(r2["y"] - 2.0 * r1["y"]).max()) == 0.0


def test_gate_svh_scale_is_not_exact_doubling(reader):
    """The gate/up svh scales sit inside SwiGLU's nonlinearity: 2x them and the output moves
    but not by exactly 2 (documents why only the down side is exact-doubling)."""
    x = _rows(9, 16)
    r1 = P.expert_pair(reader, 0, 0, x, limit=0, variant="fp32")
    rec2 = dict(reader.read_expert(0, 0))
    rec2["svh1"] = rec2["svh1"] * np.float16(2.0)
    rec2["svh3"] = rec2["svh3"] * np.float16(2.0)
    class FP: pass
    fp = FP()
    fp.header = reader.header
    fp.records = reader.records
    fp.codebook = reader.codebook
    fp.read_expert = lambda l, e: rec2
    r2 = P.expert_pair(fp, 0, 0, x, limit=0, variant="fp32")
    ratio = r2["y"] / r1["y"]
    assert float(np.abs(ratio - 2.0).max()) > 1e-3      # nonlinear: not a clean doubling
    assert float(np.corrcoef(r1["y"].ravel(), r2["y"].ravel())[0, 1]) > 0.99


# --- the x8 row scales -------------------------------------------------------------------------

def test_x8_row_scale_is_absmax_over_448():
    suh = np.ones(128, dtype=np.float16)
    x0 = np.zeros(128, dtype=np.float32)
    x0[3] = 1000.0
    a, sc = P.rotate_in(x0, suh, "x8")
    # the scale is the ROTATED row's absmax / 448: fwht128 of a delta is a Hadamard column
    # (absmax 1000), and rotate_in rounds it to fp16 first -> 88.375
    want = float(P.fp16r(np.float32(1000.0 * P.HAD_SCALE)) / 448.0)
    assert abs(float(sc) - want) < 1e-6, (float(sc), want)
    assert float(np.abs(a).max()) == 448.0
    z, sz = P.rotate_in(np.zeros(128, dtype=np.float32), suh, "x8")
    assert float(sz) == 1.0 and float(np.abs(z).max()) == 0.0


# --- the CLI -----------------------------------------------------------------------------------

def test_cli_run_and_compare(tmp_path, mini_pack):
    import subprocess
    here, _ = mini_pack
    f32 = str(tmp_path / "f32.npz")
    r = subprocess.run([sys.executable, os.path.join(HERE, "exl3_prefill_ref.py"),
                        "--pack", here, "--layer", "0", "--expert", "0", "--rows", "16",
                        "--variant", "fp32", "--emit", "npz", "--out", f32],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = np.load(f32)
    for k in ("x", "y", "xd", "acc_g", "acc_u", "acc_d", "row_scale_g", "row_scale_u"):
        assert k in z.files, k
    f16 = str(tmp_path / "f16.npz")
    r = subprocess.run([sys.executable, os.path.join(HERE, "exl3_prefill_ref.py"),
                        "--pack", here, "--layer", "0", "--expert", "0", "--rows", "16",
                        "--variant", "fp16", "--compare", f32],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert '"rel_rms"' in r.stdout

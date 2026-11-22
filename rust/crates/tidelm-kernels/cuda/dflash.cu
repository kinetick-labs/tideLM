// GLM-5.3-Flash's DFlash2 drafter: taps, norms + rope, the two-tap convs, SwiGLU, ring and block
// attention, the selector's top-16, and the small combiner kernels the drafter's layers need.
//
// Ported from kinetick-labs/TensorFold's zig/kernels/cuda/glm_dflash.cu (ours, itself after
// z-lab/dflash, MIT), kernel by kernel and signature by signature, so the two files can be read
// side by side. What changed here: the symbols carry a glm_df_ prefix (one shared library), the
// comments name our buffers, and the linears the drafter owns (qkv, o, gate/up, down, the selector
// projections, the head) are NOT here -- they go through this crate's bf16 GEMM, since the drafter
// is served in bf16 (2.2 GB) rather than TensorFold's 4-bit staging.
//
// Constants are the checkpoint's (models/glm53-dflash2/config.json): hidden 4096, 32 q heads, 8 kv
// heads, head_dim 128, block 8, conv kernel 2 over groups of 16 channels, sliding window 2048.
//
// The ring: one layer's K then V, [WIN][KVW] bf16 each, layers one after another, per row.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int D = 4096;
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD;  // 1024
constexpr int WIN = 2048;      // the sliding window (and the ring's slots)
constexpr int BLOCK = 8;       // tokens a draft block proposes
constexpr int GROUP = 16;      // channels sharing one dynamic tap weight
constexpr int GROUPS = D / GROUP;
constexpr int CH = 16;         // key chunks of the chunked attention (glm_df_attn_part)

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// The ring of one layer: K then V, [WIN][KVW] each, layers one after another.
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * WIN * KVW;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer) { return ringK(base, layer) + (long long)WIN * KVW; }

// Qwen3's per-head RMSNorm then rotate-half RoPE of one head (64 threads: dims d and d + 64), in
// place or to out.
__device__ __forceinline__ void normRope(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                         __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x;  // 0..63
    const float a = bf(x[d]), b = bf(x[d + 64]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / HD + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}
}  // namespace

// A target layer's residual, its 4 streams' mean (mode 0) or stream 0 (mode 1), into column block
// `slot` of the taps buffer [rows][5][D]. NOTE: the 4-stream assumption is TensorFold's reading of
// the target's residual; our forward must hand over the same [rows][4 * D] capture or the drafter
// sees a different input than it was trained on (see docs/dflash2-drafter.md, phase 1).
extern "C" __global__ void glm_df_tap(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int slot, int mode) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)r * 4 * D;
    __nv_bfloat16* o = out + (long long)r * 5 * D + (long long)slot * D;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        if (mode == 1) {
            o[d] = xr[d];
            continue;
        }
        const float a = ((bf(xr[d]) + bf(xr[D + d])) + bf(xr[2 * D + d])) + bf(xr[3 * D + d]);
        o[d] = tobf(a / 4.0f);
    }
}

// The block's queries (fused qkv, stride ldq) normed and rotated at pos[r] into q, keys into kb,
// values into vb. grid (rows, QH + KVH).
extern "C" __global__ void glm_df_block_qkv(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                            const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps,
                                            float theta, __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb,
                                            __nv_bfloat16* __restrict__ vb) {
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * QH * HD + h * HD, red);
    } else {
        const int kh = h - QH;
        normRope(row + QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * HD, red);
        const __nv_bfloat16* v = row + QH * HD + KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

// Context rows' keys and values (k | v) normed, rotated at pos[r], into rings[r] at slot pos % WIN
// of `layer`. grid (rows, KVH).
extern "C" __global__ void glm_df_ctx_kv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                         const int* __restrict__ pos, const uint64_t* __restrict__ rings, int layer,
                                         float eps, float theta) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    // `rings` holds one ring a stream, and rows are stream-major (BLOCK a stream): indexing it by
    // the row read past its end as soon as a block had more than one stream's worth of rows.
    const int s = r / BLOCK;
    const long long p = pos[r];
    const int slot = (int)(p % WIN);
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW;
    normRope(row + kh * HD, kn, eps, p, theta, ringK(rings[s], layer) + (long long)slot * KVW + kh * HD, red);
    __nv_bfloat16* vo = ringV(rings[s], layer) + (long long)slot * KVW + kh * HD;
    vo[threadIdx.x] = row[KVW + kh * HD + threadIdx.x];
    vo[threadIdx.x + 64] = row[KVW + kh * HD + threadIdx.x + 64];
}

// Two-tap dynamic causal conv over a block: y[t] = sum_o (base[o] + dyn[t][o][c / 16]) * z[t - o];
// add or store. grid (rows, D / 1024).
extern "C" __global__ void glm_df_conv(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn, int which,
                                       const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int add) {
    const int r = blockIdx.x, t = r % BLOCK;
    const __nv_bfloat16* dr = dyn + (long long)r * 4 * GROUPS + which * 2 * GROUPS;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * D + c]);
        if (t > 0) y += (bf(base[D + c]) + bf(dr[GROUPS + g])) * bf(z[(long long)(r - 1) * D + c]);
        __nv_bfloat16* o = out + (long long)r * D + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}

// SwiGLU over [gate | up] rows (width F each): bf16(bf16(silu(g)) * u). grid (rows, ceil(F / 1024)).
extern "C" __global__ void glm_df_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int F) {
    const int r = blockIdx.x;
    const __nv_bfloat16* g = gu + (long long)r * 2 * F;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < F; c += blockDim.x) {
        const float x = bf(g[c]);
        const float s = bf(tobf(x / (1.0f + expf(-x))));
        out[(long long)r * F + c] = tobf(s * bf(g[F + c]));
    }
}

// Each block's 8 rows attend to their ring's window (max(from, p - 2047) .. start - 1) and all 8
// block keys. grid (blocks, KVH).
extern "C" __global__ void __launch_bounds__(256) glm_df_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
                                                              const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings,
                                                              const long long* __restrict__ starts,
                                                              const long long* __restrict__ froms, int layer, float scale,
                                                              __nv_bfloat16* __restrict__ o) {
    const int s = blockIdx.x, kh = blockIdx.y;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    const long long n_ctx = start > lo ? start - lo : 0;
    for (long long n = 0; n < n_ctx + BLOCK; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        __nv_bfloat16* oh = o + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        const float inv = 1.0f / l[h];
        for (int i = 0; i < 4; ++i) oh[i] = tobf(acc[h][i] * inv);
    }
}

// glm_df_attn split over CH key chunks (running max, sum, fp32 output), merged by glm_df_attn_merge:
// not latency-bound, and the long-window case is what the drafter spends a draft's time in.
extern "C" __global__ void __launch_bounds__(256) glm_df_attn_part(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb, const __nv_bfloat16* __restrict__ vb,
    const uint64_t* __restrict__ rings, const long long* __restrict__ starts, const long long* __restrict__ froms, int layer,
    float scale, float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

// The chunks' partials of row r, head h into o[r][h] (bf16). grid (rows, QH), 128 threads (a thread
// a dim).
extern "C" __global__ void glm_df_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                             __nv_bfloat16* __restrict__ o) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}

// x += y (bf16 rows of D). grid (rows, D / 1024).
extern "C" __global__ void glm_df_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x)
        x[r * D + c] = tobf(bf(x[r * D + c]) + bf(y[r * D + c]));
}

// The split down's fp32 partials g[world][rows][D] summed in rank order into bf16 out[rows][D].
// Kept for world = 1 too: the caller passes one part and the loop does not run.
extern "C" __global__ void glm_df_sumn(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R, int world) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const long long i = r * D + c;
        float s = g[i];
        for (int k = 1; k < world; ++k) s += g[(long long)k * R * D + i];
        out[i] = tobf(s);
    }
}

// A row's top 16 logits as global ids and fp32 values: per-chunk part, then merge; ties keep low id.
// This is the block drafter's candidate set, one row a position of the block.
namespace {
constexpr int CHUNK_COLS = 4096;
constexpr int TOP = 16;

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// The block's best (value, index) of sv/si [n] (256 threads), the winner's slot cleared to -inf.
__device__ void takeBest(float* sv, int* si, int n, float* rv, int* ri, float* outv, int* outi) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (better(sv[j], si[j], bv, bi)) { bv = sv[j]; bi = si[j]; bs = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bs = os; }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { rv[w] = bv; ri[w] = bi; ri[8 + w] = bs; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) glm_df_top16_part(const __nv_bfloat16* __restrict__ logits, int cols, int offset,
                                                                    float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = bf(logits[(long long)r * cols + c0 + j]);
        si[j] = offset + c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

extern "C" __global__ void __launch_bounds__(256) glm_df_top16_merge(const float* __restrict__ pv, const int* __restrict__ pi,
                                                                     int chunks, int* __restrict__ out) {
    __shared__ float sv[64 * TOP];
    __shared__ int si[64 * TOP];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP];
    __shared__ int oi[TOP];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// A row's sums over each 64-column group, fp32 [rows][k / 64]: the input a 4-bit linear's kernel
// wants. The drafter is served in bf16 here, so nothing in this crate calls it today; it stays
// because a 4-bit drafter is the obvious next step for a 2.2 GB checkpoint and the port is where
// the two files stay comparable.
extern "C" __global__ void glm_df_gsum(const __nv_bfloat16* __restrict__ x, int ldx, int k, float* __restrict__ xs) {
    const int r = blockIdx.x, groups = k / 64;
    const __nv_bfloat16* xr = x + (long long)r * ldx;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float s = 0.f;
        for (int i = 0; i < 64; ++i) s += bf(xr[g * 64 + i]);
        xs[(long long)r * groups + g] = s;
    }
}

// ---- launchers: C ABI, stream first, cudaGetLastError() returned as int, no allocation ----------
//
// Same shape as glm.cu's, so the Rust FFI (src/dflash.rs) reads like the rest of the crate: the
// caller passes semantic counts and the launcher derives the grid. Block sizes are the kernels'
// own requirements: 64 threads where normRope maps dim d to threadIdx.x (d and d + 64), 128 where a
// thread owns one head dim, 256 everywhere the body strides by blockDim.x.

static inline int df_check() { return (int)cudaGetLastError(); }

extern "C" int glm_df_launch_tap(cudaStream_t st, const void* x, void* out, int rows, int slot, int mode) {
    glm_df_tap<<<dim3(rows, D / 1024), 256, 0, st>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, slot, mode);
    return df_check();
}

extern "C" int glm_df_launch_block_qkv(cudaStream_t st, const void* qkv, int ldq, const void* qn, const void* kn,
                                       const void* pos, int rows, float eps, float theta, void* q, void* kb, void* vb) {
    glm_df_block_qkv<<<dim3(rows, QH + KVH), 64, 0, st>>>((const __nv_bfloat16*)qkv, ldq, (const __nv_bfloat16*)qn,
                                                          (const __nv_bfloat16*)kn, (const int*)pos, eps, theta,
                                                          (__nv_bfloat16*)q, (__nv_bfloat16*)kb, (__nv_bfloat16*)vb);
    return df_check();
}

extern "C" int glm_df_launch_ctx_kv(cudaStream_t st, const void* kv, const void* kn, const void* pos, const void* rings,
                                    int layer, int rows, float eps, float theta) {
    glm_df_ctx_kv<<<dim3(rows, KVH), 64, 0, st>>>((const __nv_bfloat16*)kv, (const __nv_bfloat16*)kn, (const int*)pos,
                                                  (const uint64_t*)rings, layer, eps, theta);
    return df_check();
}

extern "C" int glm_df_launch_conv(cudaStream_t st, const void* z, const void* dyn, int which, const void* base, void* out,
                                  int rows, int add) {
    glm_df_conv<<<dim3(rows, D / 1024), 256, 0, st>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)dyn, which,
                                                      (const __nv_bfloat16*)base, (__nv_bfloat16*)out, add);
    return df_check();
}

extern "C" int glm_df_launch_swiglu(cudaStream_t st, const void* gu, void* out, int rows, int F) {
    glm_df_swiglu<<<dim3(rows, (F + 1023) / 1024), 256, 0, st>>>((const __nv_bfloat16*)gu, (__nv_bfloat16*)out, F);
    return df_check();
}

extern "C" int glm_df_launch_attn(cudaStream_t st, const void* q, const void* kb, const void* vb, const void* rings,
                                  const void* starts, const void* froms, int blocks, int layer, float scale, void* o) {
    glm_df_attn<<<dim3(blocks, KVH), 256, 0, st>>>((const __nv_bfloat16*)q, (const __nv_bfloat16*)kb,
                                                   (const __nv_bfloat16*)vb, (const uint64_t*)rings,
                                                   (const long long*)starts, (const long long*)froms, layer, scale,
                                                   (__nv_bfloat16*)o);
    return df_check();
}

extern "C" int glm_df_launch_attn_part(cudaStream_t st, const void* q, const void* kb, const void* vb, const void* rings,
                                       const void* starts, const void* froms, int blocks, int layer, float scale, int chunks,
                                       void* pm, void* pl, void* pacc) {
    glm_df_attn_part<<<dim3(blocks, KVH, chunks), 256, 0, st>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)kb, (const __nv_bfloat16*)vb, (const uint64_t*)rings,
        (const long long*)starts, (const long long*)froms, layer, scale, (float*)pm, (float*)pl, (float*)pacc);
    return df_check();
}

extern "C" int glm_df_launch_attn_merge(cudaStream_t st, const void* pm, const void* pl, const void* pacc, int rows, void* o) {
    glm_df_attn_merge<<<dim3(rows, QH), 128, 0, st>>>((const float*)pm, (const float*)pl, (const float*)pacc,
                                                      (__nv_bfloat16*)o);
    return df_check();
}

extern "C" int glm_df_launch_add(cudaStream_t st, void* x, const void* y, int rows) {
    glm_df_add<<<dim3(rows, D / 1024), 256, 0, st>>>((__nv_bfloat16*)x, (const __nv_bfloat16*)y);
    return df_check();
}

extern "C" int glm_df_launch_sumn(cudaStream_t st, const void* g, void* out, int rows, int world) {
    glm_df_sumn<<<dim3(rows, D / 1024), 256, 0, st>>>((const float*)g, (__nv_bfloat16*)out, rows, world);
    return df_check();
}

extern "C" int glm_df_launch_top16_part(cudaStream_t st, const void* logits, int rows, int cols, int offset, void* pv,
                                        void* pi) {
    glm_df_top16_part<<<dim3(rows, (cols + CHUNK_COLS - 1) / CHUNK_COLS), 256, 0, st>>>(
        (const __nv_bfloat16*)logits, cols, offset, (float*)pv, (int*)pi);
    return df_check();
}

extern "C" int glm_df_launch_top16_merge(cudaStream_t st, const void* pv, const void* pi, int rows, int chunks, void* out) {
    glm_df_top16_merge<<<dim3(rows), 256, 0, st>>>((const float*)pv, (const int*)pi, chunks, (int*)out);
    return df_check();
}

extern "C" int glm_df_launch_gsum(cudaStream_t st, const void* x, int ldx, int rows, int k, void* xs) {
    glm_df_gsum<<<dim3(rows), 128, 0, st>>>((const __nv_bfloat16*)x, ldx, k, (float*)xs);
    return df_check();
}

// The same chunked top-16 over an fp32 logits row: the target's head is fp32 here (the drafter's
// block is scored by the target, which is where the logits already land).
extern "C" __global__ void __launch_bounds__(256) glm_df_top16_part_f32(const float* __restrict__ logits, int cols,
                                                                       int offset, float* __restrict__ pv,
                                                                       int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = logits[(long long)r * cols + c0 + j];
        si[j] = offset + c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

extern "C" int glm_df_launch_top16_part_f32(cudaStream_t st, const void* logits, int rows, int cols, int offset, void* pv,
                                            void* pi) {
    glm_df_top16_part_f32<<<dim3(rows, (cols + CHUNK_COLS - 1) / CHUNK_COLS), 256, 0, st>>>(
        (const float*)logits, cols, offset, (float*)pv, (int*)pi);
    return df_check();
}

// The candidate selector, one block a stream. The chain is sequential -- the next step's
// predecessor is this step's pick -- so this is one launch a step, over the streams' current row.
// pg = predecessor_codebook[prev] * g over the ranks; each candidate scores its successor row,
// value + pg . succ[id], and the winner is the draft.
extern "C" __global__ void __launch_bounds__(256) glm_df_select(const __nv_bfloat16* __restrict__ pred,
                                                                const __nv_bfloat16* __restrict__ succ,
                                                                const __nv_bfloat16* __restrict__ g,
                                                                const int* __restrict__ cand,
                                                                const int* __restrict__ prev,
                                                                const int* __restrict__ rsel, int* __restrict__ out,
                                                                int rank) {
    __shared__ float pg[256];
    __shared__ float sc[TOP];
    const int s = blockIdx.x, r = rsel[s], p = prev[s];
    const __nv_bfloat16* gr = g + (long long)r * rank;
    const __nv_bfloat16* pr = pred + (long long)p * rank;
    for (int c = threadIdx.x; c < rank; c += blockDim.x) pg[c] = bf(pr[c]) * bf(gr[c]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int i = warp; i < TOP; i += nw) {
        const int id = cand[r * 2 * TOP + i];
        const __nv_bfloat16* sr = succ + (long long)id * rank;
        float a = 0.f;
        for (int c = lane; c < rank; c += 32) a += pg[c] * bf(sr[c]);
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) sc[i] = a + __int_as_float(cand[r * 2 * TOP + TOP + i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int i = 1; i < TOP; ++i)
            if (sc[i] > sc[best]) best = i;
        out[s] = cand[r * 2 * TOP + best];
    }
}

extern "C" int glm_df_launch_select(cudaStream_t st, const void* pred, const void* succ, const void* g, const void* cand,
                                    const void* prev, const void* rsel, void* out, int streams, int rank) {
    glm_df_select<<<dim3(streams), 256, 0, st>>>((const __nv_bfloat16*)pred, (const __nv_bfloat16*)succ,
                                                (const __nv_bfloat16*)g, (const int*)cand, (const int*)prev,
                                                (const int*)rsel, (int*)out, rank);
    return df_check();
}

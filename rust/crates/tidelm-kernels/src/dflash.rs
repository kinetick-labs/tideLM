//! FFI to `cuda/dflash.cu`, the DFlash2 drafter's device code (TensorFold's
//! `zig/kernels/cuda/glm_dflash.cu`, ported; see docs/dflash2-drafter.md).
//!
//! The drafter itself -- its weights, the ring, the taps buffer, the 5-layer forward and the
//! selector -- lives in `tidelm-glm`, which calls these launchers; this module is only the ABI.
//! Every launcher returns `cudaGetLastError()` as an int and takes the stream first, like the rest
//! of the crate. Sizes are semantic (`rows`, `blocks`, `F`), and the launcher derives the grid, so
//! the caller never encodes a block size the kernel needs (64 where one thread owns dim d and
//! d + 64, 128 where a thread owns one head dim).
//!
//! Buffers, all bf16 unless said otherwise, rows = positions the call covers:
//!
//! * taps `[rows][5][D]` — the target's residuals, one slot per `target_layer_ids` entry;
//! * q `[rows][QH * HD]`, kb/vb `[rows][KVW]` — one block's own queries and keys/values;
//! * rings — per row a device pointer (uint64) to that row's `[layers][2][WIN][KVW]` ring;
//! * pm/pl/pacc — the chunked attention's partials, `[rows][QH][CH]` and `[rows][QH][CH][HD]` f32;
//! * pv/pi — the selector's candidates, `[rows][chunks][16]` and `[rows][chunks][16]` int ids;
//! * out of `glm_df_launch_top16_merge`: `[rows][2][16]` int, ids then the values as f32 bits.

use crate::CudaStream;
use std::os::raw::{c_int, c_void};

extern "C" {
    pub fn glm_df_launch_tap_ring(stream: CudaStream, x: *const c_void, pos: *const c_void, out: *mut c_void,
                                  rows: c_int, base: c_int, slot: c_int, win: c_int) -> c_int;
    /// Row r's closed residual into `wins[lane[r]]`, at `pos[r] % WIN`, tap column `slot`.
    pub fn glm_df_launch_tap_lanes(stream: CudaStream, x: *const c_void, pos: *const c_void, lane: *const c_void,
                                   wins: *const c_void, slot: c_int, rows: c_int, nwins: c_int) -> c_int;
    /// `cin[r] = win[(start + r) % WIN]`, `n` rows of `[5][D]`.
    pub fn glm_df_launch_win_load(stream: CudaStream, win: *const c_void, start: c_int, cin: *mut c_void, n: c_int)
        -> c_int;
    pub fn glm_df_launch_tap(stream: CudaStream, x: *const c_void, out: *mut c_void, rows: c_int, slot: c_int,
                             mode: c_int) -> c_int;
    pub fn glm_df_launch_block_qkv(stream: CudaStream, qkv: *const c_void, ldq: c_int, qn: *const c_void,
                                   kn: *const c_void, pos: *const c_void, rows: c_int, eps: f32, theta: f32,
                                   q: *mut c_void, kb: *mut c_void, vb: *mut c_void) -> c_int;
    pub fn glm_df_launch_ctx_kv(stream: CudaStream, kv: *const c_void, kn: *const c_void, pos: *const c_void,
                                rings: *const c_void, layer: c_int, rows: c_int, eps: f32, theta: f32) -> c_int;
    pub fn glm_df_launch_conv(stream: CudaStream, z: *const c_void, dyn_w: *const c_void, which: c_int,
                              base: *const c_void, out: *mut c_void, rows: c_int, add: c_int) -> c_int;
    pub fn glm_df_launch_swiglu(stream: CudaStream, gu: *const c_void, out: *mut c_void, rows: c_int, f: c_int)
        -> c_int;
    pub fn glm_df_launch_attn(stream: CudaStream, q: *const c_void, kb: *const c_void, vb: *const c_void,
                              rings: *const c_void, starts: *const c_void, froms: *const c_void, blocks: c_int,
                              layer: c_int, scale: f32, o: *mut c_void) -> c_int;
    pub fn glm_df_launch_attn_part(stream: CudaStream, q: *const c_void, kb: *const c_void, vb: *const c_void,
                                   rings: *const c_void, starts: *const c_void, froms: *const c_void, blocks: c_int,
                                   layer: c_int, scale: f32, chunks: c_int, pm: *mut c_void, pl: *mut c_void,
                                   pacc: *mut c_void) -> c_int;
    pub fn glm_df_launch_attn_merge(stream: CudaStream, pm: *const c_void, pl: *const c_void, pacc: *const c_void,
                                    rows: c_int, o: *mut c_void) -> c_int;
    pub fn glm_df_launch_add(stream: CudaStream, x: *mut c_void, y: *const c_void, rows: c_int) -> c_int;
    pub fn glm_df_launch_sumn(stream: CudaStream, g: *const c_void, out: *mut c_void, rows: c_int, world: c_int)
        -> c_int;
    pub fn glm_df_launch_top16_part(stream: CudaStream, logits: *const c_void, rows: c_int, cols: c_int,
                                    offset: c_int, pv: *mut c_void, pi: *mut c_void) -> c_int;
    pub fn glm_df_launch_top16_merge(stream: CudaStream, pv: *const c_void, pi: *const c_void, rows: c_int,
                                     chunks: c_int, out: *mut c_void) -> c_int;
    pub fn glm_df_launch_gsum(stream: CudaStream, x: *const c_void, ldx: c_int, rows: c_int, k: c_int,
                              xs: *mut c_void) -> c_int;
    /// The same chunked top-16 over an fp32 logits row (the target's head is fp32).
    pub fn glm_df_launch_top16_part_f32(stream: CudaStream, logits: *const c_void, rows: c_int, cols: c_int,
                                        offset: c_int, pv: *mut c_void, pi: *mut c_void) -> c_int;
    /// One step of the selector's walk: `streams` active rows (`rsel[i]` is the row `i` reads its
    /// candidates from), `prev[i]` the chain's last id, `out[i]` the pick. One launch a step
    /// because the next step's predecessor is this step's pick.
    pub fn glm_df_launch_select(stream: CudaStream, pred: *const c_void, succ: *const c_void, g: *const c_void,
                                cand: *const c_void, prev: *const c_void, rsel: *const c_void, out: *mut c_void,
                                streams: c_int, rank: c_int) -> c_int;
}

/// The drafter's fixed sizes, from the checkpoint's `config.json`. Anything that reads them from a
/// config at runtime gets them from `tidelm-glm`'s drafter module instead; these are for the
/// kernels' callers that must size a buffer before a checkpoint is open.
pub const D: usize = 4096;
pub const HD: usize = 128;
pub const KVH: usize = 8;
pub const QH: usize = 32;
pub const KVW: usize = KVH * HD;
pub const WIN: usize = 2048;
pub const BLOCK: usize = 8;
pub const GROUP: usize = 16;
pub const CH: usize = 16;
/// Tap slots: `target_layer_ids` in `config.json` (5 for this checkpoint).
pub const TAPS: usize = 5;

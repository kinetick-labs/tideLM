//! The DFlash2 drafter, ready to serve: its weights, its block pass, and the selector's walk.
//!
//! It lives in this crate rather than in a model crate because both of them need it -- the bench
//! in `tidelm-glm` and the serving loop in `tidelm-engine` -- and only `tidelm-kernels` is below
//! both. The device work is `cuda/dflash.cu`; the linears are `blas`, the drafter being served in
//! bf16.

//! The DFlash2 drafter's weights and buffers, loaded from `models/glm53-dflash2`.

/// Errors here are messages: the drafter's failures are a missing tensor, a shape that disagrees
/// with the checkpoint, or a CUDA launcher's status, and none of them wants a taxonomy.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;


//
// This is the data side only: `dflash_fwd.rs` runs the five layers through the kernels in
// `tidelm-kernels/cuda/dflash.cu`. Everything here is bf16, as published (2.2 GB), because the
// drafter's cost is its own forward, not its bytes -- quantising it is a later step with its own
// acceptance measurement (docs/dflash2-drafter.md).
//
// Two tensors do not exist in the checkpoint and must not be invented: there is no `lm_head` (the
// block scores through the *target's* head) and no `embed_tokens` (the anchor's embedding comes
// from the target's embedding table). Three more are stored apart and served concatenated, because
// the kernels read them as one row block: `self_attn.{q,k,v}_proj` as `qkv` `[6144][4096]` (q then
// k then v, the fused layout `glm_df_block_qkv` walks), `self_attn.{k,v}_proj` alone as `kv`
// `[2048][4096]` for the context ingest, and `mlp.{gate,up}_proj` as `gu` `[24576][4096]`, the
// `[gate | up]` rows `glm_df_swiglu` splits at F.
//
// Shapes below come from the checkpoint's safetensors header (81 tensors), not from the config:
// `fc` [4096][20480] = five taps x 4096, `candidate_selector.hidden_projection` [256][4096],
// `{predecessor,successor}_codebook` [154880][256], `attention_conv.kernel_projection` and
// `mlp_conv.kernel_projection` [1024][4096] = [which][tap][256 groups], and
// `*_conv.base_kernel` [2][2][4096] = [which][tap][channel].

use tidelm_cuda::DeviceBuffer;
use tidelm_kernels::dflash::{D, KVW};



/// Positions a draft block carries: `BLOCK` from the checkpoint (`block_size: 8`).
pub const BLOCK: usize = tidelm_kernels::dflash::BLOCK;
/// The drafter's backbone layers (`num_hidden_layers: 5`).
pub const LAYERS: usize = 5;
/// `intermediate_size`.
pub const FF: usize = 12288;
/// The selector's rank (`selector_rank`).
pub const RANK: usize = 256;
/// The target's vocabulary (`draft_vocab_size` and the target's `vocab_size` agree).
pub const VOCAB: usize = 154880;
/// Rows a drafter call can carry: the verify block, which is what a round drafts for.
pub const MAX_ROWS: usize = 8;

/// The drafter's attention window, in keys. The seed reaches for 8 of them today (see
/// `seed_taps`); the checkpoint was trained on the full window.
pub const WIN: usize = tidelm_kernels::dflash::WIN;

/// One layer's weights, laid out the way the kernels index them.
pub struct DraftLayer {
    pub in_norm: DeviceBuffer<u16>,
    pub post_norm: DeviceBuffer<u16>,
    /// q_norm, k_norm: `[HD]` each, applied per head before RoPE.
    pub qn: DeviceBuffer<u16>,
    pub kn: DeviceBuffer<u16>,
    /// `[2][2][D]`: the static half of the two-tap conv, which=0 before the projection and
    /// which=1 on the projection's output (the kernel's `base + 2 * D * 2`).
    pub a_base: DeviceBuffer<u16>,
    pub m_base: DeviceBuffer<u16>,
    /// `[6144][4096]` = q | k | v, and `[24576][4096]` = gate | up.
    pub qkv: DeviceBuffer<u16>,
    pub gu: DeviceBuffer<u16>,
    /// `[2048][4096]` = k | v, the context ingest's projection.
    pub kv: DeviceBuffer<u16>,
    pub o: DeviceBuffer<u16>,
    pub down: DeviceBuffer<u16>,
    /// The dynamic conv weights' projections: `[1024][4096]`, i.e. [which][tap][group].
    pub a_proj: DeviceBuffer<u16>,
    pub m_proj: DeviceBuffer<u16>,
}

/// The whole drafter: the tap fusion, the norms, the five layers and the selector's tables.
pub struct Drafter {
    /// `[4096][20480]`: the five taps (5 x 4096) into the drafter's width.
    pub fc: DeviceBuffer<u16>,
    pub hnorm: DeviceBuffer<u16>,
    pub norm: DeviceBuffer<u16>,
    /// `[256][4096]`: the hidden the selector's codebooks are combined with.
    pub hproj: DeviceBuffer<u16>,
    /// `[154880][256]`: the previous draft's vector and the candidate's, per vocabulary entry.
    pub pred: DeviceBuffer<u16>,
    pub succ: DeviceBuffer<u16>,
    pub layers: Vec<DraftLayer>,
    /// The ring's row stride: `[layers][2][WIN][KVW]` bf16, one row's whole window.
    pub ring_row: usize,
}

impl Drafter {
    /// Load every tensor. `ck` is opened on the drafter's own directory: it is a separate
    /// checkpoint from the target's, with its own `model.safetensors` and no shard index.
    pub fn load(ck: &tidelm_model::Checkpoint) -> Result<Self> {
        let one = |name: &str| -> Result<DeviceBuffer<u16>> {
            bf16(ck, name)
        };
        // Rows of several tensors into one buffer, in the order the kernels read them.
        let stack = |parts: &[(&str, usize)]| -> Result<DeviceBuffer<u16>> {
            let mut host: Vec<u16> = Vec::new();
            for (name, rows) in parts {
                let b = bf16(ck, name)?;
                let v = b.download().map_err(|e| format!("{name}: download: {e}"))?;
                if v.len() != rows * D {
                    return Err(format!("{name}: {} values, expected {} x {D}", v.len(), rows).into());
                }
                host.extend_from_slice(&v);
            }
            DeviceBuffer::from_slice(&host).map_err(|e| format!("stack: {e}").into())
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let p = format!("layers.{l}.");
            let a = format!("{p}self_attn.");
            layers.push(DraftLayer {
                in_norm: one(&format!("{p}input_layernorm.weight"))?,
                post_norm: one(&format!("{p}post_attention_layernorm.weight"))?,
                qn: one(&format!("{a}q_norm.weight"))?,
                kn: one(&format!("{a}k_norm.weight"))?,
                a_base: one(&format!("{p}attention_conv.base_kernel"))?,
                m_base: one(&format!("{p}mlp_conv.base_kernel"))?,
                qkv: stack(&[
                    (&format!("{a}q_proj.weight"), D),
                    (&format!("{a}k_proj.weight"), KVW),
                    (&format!("{a}v_proj.weight"), KVW),
                ])?,
                kv: stack(&[(&format!("{a}k_proj.weight"), KVW), (&format!("{a}v_proj.weight"), KVW)])?,
                gu: stack(&[(&format!("{p}mlp.gate_proj.weight"), FF), (&format!("{p}mlp.up_proj.weight"), FF)])?,
                o: one(&format!("{a}o_proj.weight"))?,
                down: one(&format!("{p}mlp.down_proj.weight"))?,
                a_proj: one(&format!("{p}attention_conv.kernel_projection.weight"))?,
                m_proj: one(&format!("{p}mlp_conv.kernel_projection.weight"))?,
            });
        }
        Ok(Self {
            fc: one("fc.weight")?,
            hnorm: one("hidden_norm.weight")?,
            norm: one("norm.weight")?,
            hproj: one("candidate_selector.hidden_projection.weight")?,
            pred: one("candidate_selector.predecessor_codebook")?,
            succ: one("candidate_selector.successor_codebook")?,
            layers,
            ring_row: LAYERS * 2 * tidelm_kernels::dflash::WIN * KVW,
        })
    }

    /// Bytes of weights, for the boot log: the published 2.2 GB should come out here, and a
    /// mismatch means a tensor was read short (the stack helper checks its rows, the rest are
    /// single tensors).
    pub fn nbytes(&self) -> usize {
        let l: usize = self
            .layers
            .iter()
            .map(|l| {
                let v = |b: &DeviceBuffer<u16>| b.len();
                v(&l.in_norm) + v(&l.post_norm) + v(&l.qn) + v(&l.kn) + v(&l.a_base) + v(&l.m_base) + v(&l.qkv)
                    + v(&l.gu) + v(&l.kv) + v(&l.o) + v(&l.down) + v(&l.a_proj) + v(&l.m_proj)
            })
            .sum();
        2 * (l + self.fc.len() + self.hnorm.len() + self.norm.len() + self.hproj.len() + self.pred.len()
            + self.succ.len())
    }

    /// The ring a drafter call needs for `rows` streams: the sliding window's K and V a layer, five
    /// layers, one row per stream.
    pub fn ring_bytes(&self, rows: usize) -> usize {
        2 * rows * self.ring_row
    }
}

// The DFlash2 drafter's forward: the block pass and the context ingest, five layers each.
//
// Order and buffers follow TensorFold's `dflash_block.zig` (docs/dflash2-drafter.md has the
// sequence and the checkpoint's tensor list); the kernels are `tidelm-kernels/cuda/dflash.cu` and
// the linears are the crate's bf16 GEMM, since the drafter is served in bf16.
//
// A block is `streams * BLOCK` rows: 8 positions a stream, position 0 the anchor (the last token
// the target accepted) and 1..8 the mask token. The pass ends with `norm`, the block's hidden
// `hn` and the selector's 256-wide projection `g`; the head that turns `hn` into logits is the
// *target's* (the checkpoint has no `lm_head`), so it is called by the caller that owns it.
//
// The conv nesting is the part to read twice: each layer normalises, projects the dynamic
// weights, convolves the *normalised input* with which = 0, projects, and after the projection
// convolves again with which = 1 and adds that to the residual. One `dyn` buffer serves both
// halves, which is why it is `[M][4 * 256]`: two taps x two places.

use std::os::raw::c_void;

use tidelm_cuda::Stream;

use tidelm_kernels::blas::{Blas, Mat, R_16BF};
use tidelm_kernels::dflash as k;
use tidelm_kernels::glm;



/// Q | K | V rows of one layer's fused projection (4096 + 1024 + 1024).
pub const QKV: usize = k::QH * k::HD + 2 * k::KVW;
/// The dynamic conv weights a row carries: [which][tap][group], 2 * 2 * 256 channels/16.
pub const DYN: usize = 4 * (k::D / k::GROUP);
/// gate | up rows.
pub const GU: usize = 2 * FF;
/// The attention scale the drafter was trained with: 1/sqrt(head_dim).
pub const SCALE: f32 = 0.08838834764831845;
/// The backbone's RMSNorm epsilon (`rms_norm_eps` in the checkpoint's config).
pub const EPS: f32 = 1e-5;
/// RoPE's base (`rope_theta`).
pub const THETA: f32 = 10000.0;

fn np<T: Copy>(b: &DeviceBuffer<T>) -> *mut c_void {
    b.raw()
}

/// One drafter call's device buffers, reused round to round.
pub struct Run {
    /// The five taps, `[rows][5 * D]`: the target's capture, written by the caller.
    pub cin: DeviceBuffer<u16>,
    pub x: DeviceBuffer<u16>,
    pub y: DeviceBuffer<u16>,
    pub y2: DeviceBuffer<u16>,
    pub dynw: DeviceBuffer<u16>,
    pub qkv: DeviceBuffer<u16>,
    pub q: DeviceBuffer<u16>,
    pub kb: DeviceBuffer<u16>,
    pub vb: DeviceBuffer<u16>,
    pub att: DeviceBuffer<u16>,
    pub a: DeviceBuffer<u16>,
    pub gu: DeviceBuffer<u16>,
    pub act: DeviceBuffer<u16>,
    pub hn: DeviceBuffer<u16>,
    pub g: DeviceBuffer<u16>,
    /// The chunked attention's partials: max and sum `[M][QH][CH]`, accumulators `[M][QH][CH][HD]`.
    pub pm: DeviceBuffer<f32>,
    pub pl: DeviceBuffer<f32>,
    pub pacc: DeviceBuffer<f32>,
    /// One ring per stream (`[layers][2][WIN][KVW]` bf16 each), its per-row base pointers, and the
    /// block's tables.
    pub ring: DeviceBuffer<u16>,
    pub rings: DeviceBuffer<u64>,
    pub starts: DeviceBuffer<i64>,
    pub froms: DeviceBuffer<i64>,
    pub bpos: DeviceBuffer<i32>,
    /// The residuals of the last WIN rows of the sequence at the five tapped layers, `[WIN][5][D]`:
    /// 84 MB. The seed reads this instead of the block's eight rows; without it the drafter sees
    /// eight keys of context and accepts what eight keys can justify (0,64 a round measured).
    pub win: DeviceBuffer<u16>,
    /// Rows the capture has written, so the seed knows how much of the ring is real.
    pub win_rows: std::sync::atomic::AtomicUsize,
    /// Where `ctx_kv` writes the short seed: BLOCK positions before the anchor.
    pub bseed: DeviceBuffer<i32>,
    pub rows: usize,
    pub streams: usize,
    /// Ring elements a stream's window uses ([layers][2][WIN][KVW]).
    pub ring_row: usize,
}

impl Run {
    /// `rows` = streams * BLOCK.
    pub fn new(dr: &Drafter, rows: usize) -> Result<Self> {
        let (m, streams) = (rows, rows / BLOCK);
        Ok(Self {
            cin: DeviceBuffer::zeroed(m * 5 * k::D)?,
            x: DeviceBuffer::zeroed(m * k::D)?,
            y: DeviceBuffer::zeroed(m * k::D)?,
            y2: DeviceBuffer::zeroed(m * k::D)?,
            dynw: DeviceBuffer::zeroed(m * DYN)?,
            qkv: DeviceBuffer::zeroed(m * QKV)?,
            q: DeviceBuffer::zeroed(m * k::QH * k::HD)?,
            kb: DeviceBuffer::zeroed(m * k::KVW)?,
            vb: DeviceBuffer::zeroed(m * k::KVW)?,
            att: DeviceBuffer::zeroed(m * k::QH * k::HD)?,
            a: DeviceBuffer::zeroed(m * k::D)?,
            gu: DeviceBuffer::zeroed(m * GU)?,
            act: DeviceBuffer::zeroed(m * FF)?,
            hn: DeviceBuffer::zeroed(m * k::D)?,
            g: DeviceBuffer::zeroed(m * RANK)?,
            pm: DeviceBuffer::zeroed(m * k::QH * k::CH)?,
            pl: DeviceBuffer::zeroed(m * k::QH * k::CH)?,
            pacc: DeviceBuffer::zeroed(m * k::QH * k::CH * k::HD)?,
            ring: DeviceBuffer::zeroed(streams * dr.ring_row)?,
            rings: DeviceBuffer::zeroed(streams)?,
            starts: DeviceBuffer::zeroed(streams)?,
            froms: DeviceBuffer::zeroed(streams)?,
            bpos: DeviceBuffer::zeroed(m)?,
            bseed: DeviceBuffer::zeroed(BLOCK)?,
            win: DeviceBuffer::zeroed(WIN * 5 * k::D)?,
            win_rows: std::sync::atomic::AtomicUsize::new(0),
            rows,
            streams,
            ring_row: dr.ring_row,
        })
    }

    /// A run over `rows` consecutive positions of a window, all sharing ONE ring. The block's own
    /// run gives a ring a stream and a stream BLOCK rows (`ring[s]`, s = r / BLOCK), which is right
    /// for a verify pass -- eight rows, one context -- and wrong for the seed: 512 rows there are
    /// 64 groups over a single ring, the 2048 keys of the window, not 64 rings and 2,5 GiB.
    ///
    /// Nothing new is allocated for K/V: `set_seed_tables` points every group at the ring the block
    /// already reads, so the seed writes exactly what the next attention will look up.
    pub fn new_seed(dr: &Drafter, rows: usize) -> Result<Self> {
        let (m, groups) = (rows, rows / BLOCK);
        Ok(Self {
            cin: DeviceBuffer::zeroed(m * 5 * k::D)?,
            x: DeviceBuffer::zeroed(m * k::D)?,
            y: DeviceBuffer::zeroed(m * k::D)?,
            y2: DeviceBuffer::zeroed(m * k::D)?,
            dynw: DeviceBuffer::zeroed(m * DYN)?,
            qkv: DeviceBuffer::zeroed(m * QKV)?,
            q: DeviceBuffer::zeroed(m * k::QH * k::HD)?,
            kb: DeviceBuffer::zeroed(m * k::KVW)?,
            vb: DeviceBuffer::zeroed(m * k::KVW)?,
            att: DeviceBuffer::zeroed(m * k::QH * k::HD)?,
            a: DeviceBuffer::zeroed(m * k::D)?,
            gu: DeviceBuffer::zeroed(m * GU)?,
            act: DeviceBuffer::zeroed(m * FF)?,
            hn: DeviceBuffer::zeroed(m * k::D)?,
            g: DeviceBuffer::zeroed(m * RANK)?,
            pm: DeviceBuffer::zeroed(m * k::QH * k::CH)?,
            pl: DeviceBuffer::zeroed(m * k::QH * k::CH)?,
            pacc: DeviceBuffer::zeroed(m * k::QH * k::CH * k::HD)?,
            ring: DeviceBuffer::zeroed(1)?,
            rings: DeviceBuffer::zeroed(groups)?,
            starts: DeviceBuffer::zeroed(groups)?,
            froms: DeviceBuffer::zeroed(groups)?,
            bpos: DeviceBuffer::zeroed(m)?,
            bseed: DeviceBuffer::zeroed(BLOCK)?,
            win: DeviceBuffer::zeroed(1)?,
            win_rows: std::sync::atomic::AtomicUsize::new(0),
            rows,
            streams: groups,
            ring_row: dr.ring_row,
        })
    }

    /// `base` is the position of row 0; group g starts at `base + g * BLOCK` and may read the ring
    /// from `lo`. Every group's `rings` entry is the same pointer: one ring, many rows.
    pub fn set_seed_tables(&self, st: &Stream, ring: *mut c_void, base: i64, lo: i64) -> Result<()> {
        let g = self.streams;
        let starts: Vec<i64> = (0..g).map(|s| base + (s * BLOCK) as i64).collect();
        let froms: Vec<i64> = vec![lo; g];
        let rings: Vec<u64> = vec![ring as u64; g];
        let bpos: Vec<i32> = (0..self.rows).map(|r| (base + r as i64) as i32).collect();
        let up = |buf: &DeviceBuffer<i64>, v: &[i64]| -> Result<()> {
            buf.upload_async(0, v, st).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))
        };
        up(&self.starts, &starts)?;
        up(&self.froms, &froms)?;
        self.rings.upload_async(0, &rings, st).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;
        self.bpos.upload_async(0, &bpos, st).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;
        Ok(())
    }

    /// The tables a pass reads: per stream its position (the anchor's), the start of its context
    /// in the ring and how far back the window may look; and per row its position.
    /// `bseed` is where the short seed writes its keys -- the BLOCK positions *before* the anchor,
    /// which is the window attention actually reads (`[from, p0)`). Writing the seed at the rows'
    /// own positions (p0..p0+7) put every key outside that window: the pass then read n_ctx zeroed
    /// slots instead, and `exp(0 - m)` for two thousand of them drowned the eight real ones.
    pub fn set_tables(&self, st: &Stream, starts: &[i64], froms: &[i64], bpos: &[i64], bseed: &[i64]) -> Result<()> {
        fn put<T: Copy + 'static>(buf: &DeviceBuffer<T>, st: &Stream, data: &[T]) -> Result<()> {
            // Straight into the table: `from_slice` + `copy_from` allocated and freed ~1 MiB a
            // call, five times a round, and `cudaFree` synchronizes the device -- that, not the two
            // stream joins, is where the round's 190 ms went against the bench's 9.4.
            if data.len() != buf.len() {
                return Err(format!("dflash tables: {} values for {} rows", data.len(), buf.len()).into());
            }
            buf.upload_async(0, data, st)
                .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))
        }
        put(&self.starts, st, starts)?;
        put(&self.froms, st, froms)?;
        let b: Vec<i32> = bpos.iter().map(|&p| p as i32).collect();
        put(&self.bpos, st, &b)?;
        let sd: Vec<i32> = bseed.iter().map(|&p| p as i32).collect();
        put(&self.bseed, st, &sd)?;
        // A ring per stream, contiguous: stream s owns ring.row elements.
        let mut r: Vec<u64> = Vec::with_capacity(self.streams);
        for s in 0..self.streams {
            r.push(self.ring.raw() as u64 + (s * self.ring_row * 2) as u64);
        }
        put(&self.rings, st, &r)?;
        Ok(())
    }

    /// Capture at the layer whose residual is `x` (`[rows][4][D]`, the four streams) into tap
    /// slot `slot`: the mean of the streams, which is the layout `fc` reads.
    pub fn tap(&self, st: &Stream, x: *const c_void, slot: usize) -> Result<()> {
        ck(unsafe { k::glm_df_launch_tap(st.raw(), x, np(&self.cin), self.rows as i32, slot as i32, 0) }, "tap")
    }

    /// The same tap, into the window ring, at each row's own position (`pos[base + r]`). Only the
    /// prefill has WIN rows of residuals at once; a verify pass has eight.
    pub fn tap_window(&self, st: &Stream, x: *const c_void, pos: *const c_void, slot: usize, rows: usize, base: usize) -> Result<()> {
        ck(unsafe {
            k::glm_df_launch_tap_ring(st.raw(), x, pos, np(&self.win), rows as i32, base as i32, slot as i32, WIN as i32)
        }, "tap ring")?;
        self.win_rows.fetch_max(base + rows, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// `x = rmsnorm(fc(cin), hidden_norm)`: the five taps into the drafter's width.
    pub fn embed_taps(&self, st: &Stream, blas: &Blas, dr: &Drafter) -> Result<()> {
        self.gemm(blas, &self.cin, &dr.fc, k::D, 5 * k::D, &self.y2)?;
        self.mark(st, &self.cin, "embed cin")?;
        self.mark(st, &self.y2, "embed fc")?;
        unsafe {
            glm::glm_rmsnorm(
                st.raw(),
                np(&self.y2),
                k::D as i64,
                std::ptr::null(),
                0,
                np(&dr.hnorm),
                np(&self.x),
                k::D as i64,
                self.rows as i32,
                k::D as i32,
                EPS,
            );
        }
        Ok(())
    }

    /// The five layers over `rows` (the taps already in `cin`), then `norm` and the selector's
    /// projection. Returns nothing: `hn` and `g` are the outputs.
    pub fn pass(&self, st: &Stream, blas: &Blas, dr: &Drafter) -> Result<()> {
        self.pass_n(st, blas, dr, LAYERS)
    }

    /// One discarded pass: the weights' first touch is 2.2 GB of DRAM, and a timing that includes
    /// it measures the load, not the arithmetic. Not a correctness step -- the pass is bit-for-bit
    /// reproducible without it, which is how the stream bug below was found.
    pub fn warm(&self, st: &Stream, blas: &Blas, dr: &Drafter) -> Result<()> {
        for _ in 0..1 {
            self.embed_taps(st, blas, dr)?;
            self.pass(st, blas, dr)?;
        }
        st.synchronize()?;
        Ok(())
    }

    /// `pass` over the first `n` layers only: the bisection `pass`'s determinism needs, and the
    /// only way to say which layer first reads something nobody wrote.
    pub fn pass_n(&self, st: &Stream, blas: &Blas, dr: &Drafter, n: usize) -> Result<()> {
        let sr = st.raw();
        let (m, d) = (self.rows, k::D);
        unsafe {
            for (li, l) in dr.layers.iter().enumerate().take(n) {
                // y = rmsnorm(x, in_norm); dyn = a_proj(y); y2 = conv(y, which 0)
                glm::glm_rmsnorm(sr, np(&self.x), d as i64, std::ptr::null(), 0, np(&l.in_norm), np(&self.y), d as i64,
                                 m as i32, d as i32, EPS);
                self.gemm(blas, &self.y, &l.a_proj, DYN, d, &self.dynw)?;
                k::glm_df_launch_conv(sr, np(&self.y), np(&self.dynw), 0, np(&l.a_base), np(&self.y2), m as i32, 0);
                if li == 0 {
                    self.mark(st, &self.y, "l0 norm")?;
                    self.mark(st, &self.dynw, "l0 dyn")?;
                    self.mark(st, &self.y2, "l0 conv0")?;
                }
                // qkv = qkv_lin(y2), then the queries and this block's keys/values
                self.gemm(blas, &self.y2, &l.qkv, QKV, d, &self.qkv)?;
                if li == 0 {
                    self.mark(st, &self.qkv, "l0 qkv")?;
                }
                k::glm_df_launch_block_qkv(sr, np(&self.qkv), QKV as i32, np(&l.qn), np(&l.kn), np(&self.bpos), m as i32,
                                           EPS, THETA, np(&self.q), np(&self.kb), np(&self.vb));
                if li == 0 {
                    self.mark(st, &self.q, "l0 q")?;
                    self.mark(st, &self.kb, "l0 kb")?;
                }
                // attention over the ring's window plus the block's own 8 keys, in 16 chunks
                k::glm_df_launch_attn_part(sr, np(&self.q), np(&self.kb), np(&self.vb), np(&self.rings), np(&self.starts),
                                           np(&self.froms), self.streams as i32, li as i32, SCALE, k::CH as i32,
                                           np(&self.pm), np(&self.pl), np(&self.pacc));
                k::glm_df_launch_attn_merge(sr, np(&self.pm), np(&self.pl), np(&self.pacc), m as i32, np(&self.att));
                if li == 0 {
                    self.mark(st, &self.att, "l0 att")?;
                }
                // a = o(att); x += conv(a, which 1)
                self.gemm(blas, &self.att, &l.o, d, k::QH * k::HD, &self.a)?;
                if li == 0 {
                    self.mark(st, &self.a, "l0 o")?;
                }
                let base1 = self.base_at(&l.a_base, 1);
                k::glm_df_launch_conv(sr, np(&self.a), np(&self.dynw), 1, base1, np(&self.x), m as i32, 1);
                if li == 0 {
                    self.mark(st, &self.x, "l0 attn out")?;
                }
                // the MLP, wrapped the same way
                glm::glm_rmsnorm(sr, np(&self.x), d as i64, std::ptr::null(), 0, np(&l.post_norm), np(&self.y), d as i64,
                                 m as i32, d as i32, EPS);
                self.gemm(blas, &self.y, &l.m_proj, DYN, d, &self.dynw)?;
                k::glm_df_launch_conv(sr, np(&self.y), np(&self.dynw), 0, np(&l.m_base), np(&self.y2), m as i32, 0);
                self.gemm(blas, &self.y2, &l.gu, GU, d, &self.gu)?;
                k::glm_df_launch_swiglu(sr, np(&self.gu), np(&self.act), m as i32, FF as i32);
                self.gemm(blas, &self.act, &l.down, d, FF, &self.a)?;
                let base1 = self.base_at(&l.m_base, 1);
                k::glm_df_launch_conv(sr, np(&self.a), np(&self.dynw), 1, base1, np(&self.x), m as i32, 1);
            }
            // hn = rmsnorm(x, norm); g = hidden_projection(hn)
            glm::glm_rmsnorm(sr, np(&self.x), d as i64, std::ptr::null(), 0, np(&dr.norm), np(&self.hn), d as i64,
                             m as i32, d as i32, EPS);
        }
        self.gemm(blas, &self.hn, &dr.hproj, RANK, d, &self.g)
    }

    /// With `DF_DUMP=1`, each intermediate's hash after the kernels of layer 0: two passes whose
    /// hashes agree mean the drift is downstream of the last label that printed.
    fn mark(&self, st: &Stream, b: &DeviceBuffer<u16>, label: &str) -> Result<()> {
        if std::env::var("DF_DUMP").as_deref() != Ok("1") {
            return Ok(());
        }
        st.synchronize()?;
        let v = b.download()?;
        let h = v.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
        eprintln!("  {label}: {h:016x} first {:?}", &v[..4]);
        Ok(())
    }

    /// y[M][N] = x[M][K] . W^T, with W the [N][K] bf16 the checkpoint stores.
    fn gemm(&self, blas: &Blas, x: &DeviceBuffer<u16>, w: &DeviceBuffer<u16>, n: usize, kk: usize,
            y: &DeviceBuffer<u16>) -> Result<()> {
        blas.gemm_rm(self.rows, n, kk, Mat { ptr: x.raw(), ty: R_16BF, ld: kk as i64 },
                     Mat { ptr: w.raw(), ty: R_16BF, ld: kk as i64 }, true, y.raw(), R_16BF, n as i64)
            .map_err(|e| {
                // The shapes a failure happened on, once: a status 13 in the serving loop and not in
                // the bench is a fact about this call, not about cublas.
                e.into()
            })

    }

    /// `base_kernel` is [which][tap][D]; the kernel's `which = 1` wants the second `2 * D` block.
    fn base_at(&self, base: &DeviceBuffer<u16>, which: usize) -> *mut c_void {
        unsafe { (base.raw() as *mut u8).add(which * 2 * k::D * 2) as *mut c_void }
    }
}

/// A block's tail: the candidates the target's head scores, and the selector's walk over them.
///
/// The head is the target's, so this struct never calls it: the engine writes the block's fp32
/// logits and hands the pointer to `candidates`. The walk is `BLOCK - 1` launches because each
/// step's predecessor is the previous step's pick; the chain lives on the device (`chain`) and only
/// the drafts come back.
pub struct Block {
    pub run: Run,
    pub pv: DeviceBuffer<f32>,
    pub pi: DeviceBuffer<i32>,
    /// `[rows][2][16]`: ids then the values as f32 bits, as `top16_merge` writes them.
    pub cand: DeviceBuffer<i32>,
    /// `[BLOCK][streams]`: the walk's ids, `chain[0]` the anchors.
    pub chain: DeviceBuffer<i32>,
    /// `[BLOCK - 1][streams]`: which row each step reads its candidates from.
    pub rsel: DeviceBuffer<i32>,
    pub chunks: usize,
    /// Last position the window was seeded at. The seed costs a drafter pass over the window; it
    /// grows by a token or three a round, so re-seeding every 64 positions is enough to keep it
    /// current at a fraction of a millisecond a step.
    pub seeded_to: std::sync::atomic::AtomicI64,
    pub streams: usize,
}

impl Block {
    pub fn new(dr: &Drafter, rows: usize) -> Result<Self> {
        let streams = rows / BLOCK;
        let chunks = VOCAB.div_ceil(4096);
        let mut rsel = Vec::with_capacity((BLOCK - 1) * streams);
        for step in 1..BLOCK {
            for s in 0..streams {
                rsel.push((s * BLOCK + step) as i32);
            }
        }
        Ok(Self {
            run: Run::new(dr, rows)?,
            pv: DeviceBuffer::zeroed(rows * chunks * 16)?,
            pi: DeviceBuffer::zeroed(rows * chunks * 16)?,
            cand: DeviceBuffer::zeroed(rows * 2 * 16)?,
            chain: DeviceBuffer::zeroed(BLOCK * streams)?,
            rsel: DeviceBuffer::from_slice(&rsel)?,
            chunks,
            seeded_to: std::sync::atomic::AtomicI64::new(-1),
            streams,
        })
    }

    /// Fill the ring from the tap rows themselves: each tapped layer's hidden projected through
    /// that layer's `k|v` and written at the row's position, which is `ctx_kv`'s job.
    ///
    /// This is the *short* seed: `BLOCK` keys, the rows the taps hold. TensorFold seeds `WIN`
    /// (2048) keys by running the drafter's own context over the prompt's tail, which needs the
    /// last 2048 rows' residual at those five layers kept from prefill -- the next piece, and the
    /// reason a draft's acceptance will read low until it exists.
    pub fn seed_taps(&self, st: &Stream, blas: &Blas, dr: &Drafter) -> Result<()> {
        let (m, d) = (self.run.rows, k::D);
        for (li, l) in dr.layers.iter().enumerate() {
            // taps slot li of every row, stride 5 * D, into the layer's k|v [rows][2 * KVW]
            blas.gemm_rm(m, 2 * k::KVW, d, Mat { ptr: at(&self.run.cin, li * d), ty: R_16BF, ld: (5 * d) as i64 },
                         Mat { ptr: l.kv.raw(), ty: R_16BF, ld: d as i64 }, true, np(&self.run.qkv), R_16BF,
                         (2 * k::KVW) as i64)
                .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;
            ck(unsafe {
                k::glm_df_launch_ctx_kv(st.raw(), np(&self.run.qkv), np(&l.kn), np(&self.run.bseed), np(&self.run.rings),
                                        li as i32, m as i32, EPS, THETA)
            }, "ctx_kv")?;
        }
        Ok(())
    }

    /// Seed the drafter's ring from the captured window: every BLOCK rows of the sequence's tail gets
    /// its own pass over the block's ring, in position order, so the keys the next attention looks up
    /// are the prompt's own state and not the eight rows of the verify step.
    ///
    /// Groups of BLOCK, not one wide pass: the block kernels give a ring a stream and a stream BLOCK
    /// rows, so a 512-row pass would allocate 64 rings (2,5 GiB). This way nothing is allocated at
    /// all -- the run and its ring already exist -- and the cost is a drafter pass a BLOCK rows.
    pub fn seed_window(&self, st: &Stream, blas: &Blas, dr: &Drafter, p0: i64) -> Result<usize> {
        use std::sync::atomic::Ordering::Relaxed;
        let captured = self.run.win_rows.load(Relaxed) as i64;
        let want = p0.min(WIN as i64).min(captured);
        if want < BLOCK as i64 {
            return Ok(0);
        }
        let last = self.seeded_to.load(Relaxed);
        if last >= 0 && p0 - last < 64 {
            return Ok(0);
        }
        let mut base = 0i64;
        let mut seeded = 0usize;
        while base + BLOCK as i64 <= want {
            // the window's rows in that slice, straight into cin
            self.run.cin.copy_from(0, &self.run.win, base as usize * 5 * k::D, BLOCK * 5 * k::D, st)
                .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;
            let starts = [base];
            let froms = [0i64];
            let bpos: Vec<i64> = (0..BLOCK as i64).map(|i| base + i).collect();
            self.run.set_tables(st, &starts, &froms, &bpos, &bpos[..BLOCK])?;
            self.run.embed_taps(st, blas, dr).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e))?;
            self.run.pass(st, blas, dr).map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e))?;
            for (li, l) in dr.layers.iter().enumerate() {
                ck(unsafe {
                    k::glm_df_launch_ctx_kv(st.raw(), np(&self.run.qkv), np(&l.kn), np(&self.run.bpos),
                                            np(&self.run.rings), li as i32, BLOCK as i32, EPS, THETA)
                }, "ctx_kv seed")?;
            }
            base += BLOCK as i64;
            seeded += BLOCK;
        }
        self.seeded_to.store(p0, Relaxed);
        Ok(seeded)
    }

    /// The candidate list from **bf16** logits, which is what the engine's head leaves in
    /// `act.logits` (`glm_argmax_bf16` reads the same buffer). Same chunked top-16, same layout.
    /// The target's head logits ([rows][vocab], f32 -- tidelm-glm's `act.head`) into the block's
    /// candidates. The f32 variant of the kernel already existed: no cast, no copy.
    pub fn candidates_bf16(&self, st: &Stream, logits: *const c_void, vocab: usize) -> Result<()> {
        ck(unsafe {
            k::glm_df_launch_top16_part_f32(st.raw(), logits, self.run.rows as i32, vocab as i32, 0,
                                            np(&self.pv), np(&self.pi))
        }, "top16 part f32")?;
        // The part kernel grids over the width it is given (vl, the target's head), while pv/pi are
        // sized on the checkpoint's VOCAB: merging `self.chunks` here read 32 never-written slots
        // per row -- value 0.0, id 0 -- and 0.0 beats every negative logit, so id 0 was a candidate
        // on every row. Merge exactly as many chunks as were written.
        let chunks = vocab.div_ceil(4096);
        if vocab > VOCAB {
            return Err(format!("dflash top16: {vocab} wide, the codebooks hold {VOCAB}").into());
        }
        ck(unsafe {
            k::glm_df_launch_top16_merge(st.raw(), np(&self.pv), np(&self.pi), self.run.rows as i32, chunks as i32,
                                         np(&self.cand))
        }, "top16 merge")
    }

    /// `round` over bf16 logits (the engine's own buffer).
    pub fn round_bf16(&self, st: &Stream, blas: &Blas, dr: &Drafter, logits: *const c_void, anchors: &[i32],
                      vocab: usize) -> Result<Vec<i32>> {
        if anchors.len() != self.streams {
            return Err(format!("round: {} anchors for {} streams", anchors.len(), self.streams).into());
        }
        self.run.embed_taps(st, blas, dr)?;
        self.run.pass(st, blas, dr)?;
        self.candidates_bf16(st, logits, vocab)?;
        self.anchor(st, anchors)?;
        self.walk(st, dr)?;
        self.drafts(st)
    }

    /// One round, from taps already in `run.cin` to the drafts the walk chose:
    /// `embed_taps -> pass -> top-16 over the target's logits -> the walk`.
    ///
    /// `logits` is the *target's* head over the same 8 rows, `[rows][VOCAB]` fp32 -- the checkpoint
    /// has no `lm_head`, and scoring the block through anything else would be a distribution the
    /// drafter never saw. `anchors` is one token a stream, the last the target accepted.
    pub fn round(&self, st: &Stream, blas: &Blas, dr: &Drafter, logits: *const c_void, anchors: &[i32])
                 -> Result<Vec<i32>> {
        if anchors.len() != self.streams {
            return Err(format!("round: {} anchors for {} streams", anchors.len(), self.streams).into());
        }
        self.run.embed_taps(st, blas, dr)?;
        self.run.pass(st, blas, dr)?;
        self.candidates(st, logits)?;
        self.anchor(st, anchors)?;
        self.walk(st, dr)?;
        self.drafts(st)
    }

    /// The candidate list of every row from the target's fp32 logits `[rows][VOCAB]`.
    pub fn candidates(&self, st: &Stream, logits: *const c_void) -> Result<()> {
        ck(unsafe {
            k::glm_df_launch_top16_part_f32(st.raw(), logits, self.run.rows as i32, VOCAB as i32, 0, np(&self.pv),
                                            np(&self.pi))
        }, "top16 part")?;
        ck(unsafe {
            k::glm_df_launch_top16_merge(st.raw(), np(&self.pv), np(&self.pi), self.run.rows as i32, self.chunks as i32,
                                         np(&self.cand))
        }, "top16 merge")
    }

    /// The anchors (`streams` ids) into `chain[0]`, then the walk. Leaves the drafts in `chain`.
    pub fn walk(&self, st: &Stream, dr: &Drafter) -> Result<()> {
        for step in 1..BLOCK {
            ck(unsafe {
                k::glm_df_launch_select(
                   st.raw(),
                   np(&dr.pred),
                   np(&dr.succ),
                   np(&self.run.g),
                   np(&self.cand),
                   at(&self.chain, (step - 1) * self.streams),
                   at(&self.rsel, (step - 1) * self.streams),
                   at(&self.chain, step * self.streams),
                   self.streams as i32,
                   RANK as i32,
                )
            }, "select")?;
        }
        Ok(())
    }

    /// The drafts the walk chose, `[BLOCK - 1][streams]` row-major: step 1 first.
    pub fn drafts(&self, st: &Stream) -> Result<Vec<i32>> {
        st.synchronize()?;
        let mut v = self.chain.download()?;
        v.drain(..self.streams);
        Ok(v)
    }

    /// The anchors go in at the head of the chain.
    pub fn anchor(&self, st: &Stream, ids: &[i32]) -> Result<()> {
        self.chain.upload_async(0, ids, st)
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))
    }
}

/// `b`'s pointer `n` elements in.
fn at<T: Copy>(b: &DeviceBuffer<T>, n: usize) -> *mut c_void {
    let sz = std::mem::size_of::<T>();
    unsafe { (b.raw() as *mut u8).add(n * sz) as *mut c_void }
}

/// A CUDA launcher's status: 0, or the last error the device reported.
fn ck(rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("dflash {what}: cuda error {rc}").into())
    }
}



/// The target layers DFlash2 reads, in tap-slot order (the checkpoint's `target_layer_ids`).
/// They are captured **after** the layer's residual is written, which is not where this engine's
/// DSpark hook reads: see docs/dflash2-drafter.md, "The tap is after the layer".
pub const TAP_LAYERS: [usize; 5] = [5, 14, 24, 33, 42];

/// Layer `l`'s tap slot, or None for a layer DFlash2 does not read.
pub fn tap_slot(l: usize) -> Option<usize> {
    TAP_LAYERS.iter().position(|&x| x == l)
}

/// A bf16 tensor from the checkpoint to the device. The drafter ships bf16 throughout, so a
/// tensor of another dtype is an error rather than something to widen: a silently cast weight is
/// how a drafter ends up scoring the wrong distribution.
fn bf16(ck: &tidelm_model::Checkpoint, name: &str) -> Result<DeviceBuffer<u16>> {
    let meta = ck.meta(name).map_err(|e| format!("{name}: {e}"))?;
    if meta.dtype != "BF16" {
        return Err(format!("{name}: {} where bf16 was expected", meta.dtype).into());
    }
    let bytes = ck.read_tensor_vec(name).map_err(|e| format!("{name}: {e}"))?.0;
    let v: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    DeviceBuffer::from_slice(&v).map_err(|e| format!("{name}: {e}").into())
}


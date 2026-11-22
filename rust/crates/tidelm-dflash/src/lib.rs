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
            rows,
            streams,
            ring_row: dr.ring_row,
        })
    }

    /// The tables a pass reads: per stream its position (the anchor's), the start of its context
    /// in the ring and how far back the window may look; and per row its position.
    pub fn set_tables(&self, st: &Stream, starts: &[i64], froms: &[i64], bpos: &[i64]) -> Result<()> {
        fn put<T: Copy>(buf: &DeviceBuffer<T>, st: &Stream, data: &[T]) -> Result<()> {
            let tmp = DeviceBuffer::from_slice(data)?;
            buf.copy_from(0, &tmp, 0, data.len(), st)?;
            Ok(())
        }
        put(&self.starts, st, starts)?;
        put(&self.froms, st, froms)?;
        let b: Vec<i32> = bpos.iter().map(|&p| p as i32).collect();
        put(&self.bpos, st, &b)?;
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
            .map_err(|e| e.into())
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
            streams,
        })
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
        let a = DeviceBuffer::from_slice(ids)?;
        self.chain.copy_from(0, &a, 0, ids.len(), st)?;
        Ok(())
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


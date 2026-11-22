//! The DFlash2 drafter's forward: the block pass and the context ingest, five layers each.
//!
//! Order and buffers follow TensorFold's `dflash_block.zig` (docs/dflash2-drafter.md has the
//! sequence and the checkpoint's tensor list); the kernels are `tidelm-kernels/cuda/dflash.cu` and
//! the linears are the crate's bf16 GEMM, since the drafter is served in bf16.
//!
//! A block is `streams * BLOCK` rows: 8 positions a stream, position 0 the anchor (the last token
//! the target accepted) and 1..8 the mask token. The pass ends with `norm`, the block's hidden
//! `hn` and the selector's 256-wide projection `g`; the head that turns `hn` into logits is the
//! *target's* (the checkpoint has no `lm_head`), so it is called by the caller that owns it.
//!
//! The conv nesting is the part to read twice: each layer normalises, projects the dynamic
//! weights, convolves the *normalised input* with which = 0, projects, and after the projection
//! convolves again with which = 1 and adds that to the residual. One `dyn` buffer serves both
//! halves, which is why it is `[M][4 * 256]`: two taps x two places.

use std::os::raw::c_void;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_kernels::blas::{Blas, Mat, R_16BF};
use tidelm_kernels::dflash as k;
use tidelm_kernels::glm;

use crate::dflash::{Drafter, BLOCK, FF, LAYERS, RANK};
use crate::Result;

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

    /// `x = rmsnorm(fc(cin), hidden_norm)`: the five taps into the drafter's width.
    pub fn embed_taps(&self, blas: &Blas, dr: &Drafter) -> Result<()> {
        self.gemm(blas, &self.cin, &dr.fc, k::D, 5 * k::D, &self.y2)?;
        unsafe {
            glm::glm_rmsnorm(
                std::ptr::null_mut(),
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

    /// The five layers over `rows` (the taps already in `x` by `fc`), then `norm` and the
    /// selector's projection. Returns nothing: `hn` and `g` are the outputs.
    pub fn pass(&self, st: &Stream, blas: &Blas, dr: &Drafter) -> Result<()> {
        let sr = st.raw();
        let (m, d) = (self.rows, k::D);
        unsafe {
            for (li, l) in dr.layers.iter().enumerate() {
                // y = rmsnorm(x, in_norm); dyn = a_proj(y); y2 = conv(y, which 0)
                glm::glm_rmsnorm(sr, np(&self.x), d as i64, std::ptr::null(), 0, np(&l.in_norm), np(&self.y), d as i64,
                                 m as i32, d as i32, EPS);
                self.gemm(blas, &self.y, &l.a_proj, DYN, d, &self.dynw)?;
                k::glm_df_launch_conv(sr, np(&self.y), np(&self.dynw), 0, np(&l.a_base), np(&self.y2), m as i32, 0);
                // qkv = qkv_lin(y2), then the queries and this block's keys/values
                self.gemm(blas, &self.y2, &l.qkv, QKV, d, &self.qkv)?;
                k::glm_df_launch_block_qkv(sr, np(&self.qkv), QKV as i32, np(&l.qn), np(&l.kn), np(&self.bpos), m as i32,
                                           EPS, THETA, np(&self.q), np(&self.kb), np(&self.vb));
                // attention over the ring's window plus the block's own 8 keys, in 16 chunks
                k::glm_df_launch_attn_part(sr, np(&self.q), np(&self.kb), np(&self.vb), np(&self.rings), np(&self.starts),
                                           np(&self.froms), self.streams as i32, li as i32, SCALE, k::CH as i32,
                                           np(&self.pm), np(&self.pl), np(&self.pacc));
                k::glm_df_launch_attn_merge(sr, np(&self.pm), np(&self.pl), np(&self.pacc), m as i32, np(&self.att));
                // a = o(att); x += conv(a, which 1)
                self.gemm(blas, &self.att, &l.o, d, k::QH * k::HD, &self.a)?;
                let base1 = self.base_at(&l.a_base, 1);
                k::glm_df_launch_conv(sr, np(&self.a), np(&self.dynw), 1, base1, np(&self.x), m as i32, 1);
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

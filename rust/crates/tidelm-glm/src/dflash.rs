//! The DFlash2 drafter's weights and buffers, loaded from `models/glm53-dflash2`.
//!
//! This is the data side only: `dflash_fwd.rs` runs the five layers through the kernels in
//! `tidelm-kernels/cuda/dflash.cu`. Everything here is bf16, as published (2.2 GB), because the
//! drafter's cost is its own forward, not its bytes -- quantising it is a later step with its own
//! acceptance measurement (docs/dflash2-drafter.md).
//!
//! Two tensors do not exist in the checkpoint and must not be invented: there is no `lm_head` (the
//! block scores through the *target's* head) and no `embed_tokens` (the anchor's embedding comes
//! from the target's embedding table). Three more are stored apart and served concatenated, because
//! the kernels read them as one row block: `self_attn.{q,k,v}_proj` as `qkv` `[6144][4096]` (q then
//! k then v, the fused layout `glm_df_block_qkv` walks), `self_attn.{k,v}_proj` alone as `kv`
//! `[2048][4096]` for the context ingest, and `mlp.{gate,up}_proj` as `gu` `[24576][4096]`, the
//! `[gate | up]` rows `glm_df_swiglu` splits at F.
//!
//! Shapes below come from the checkpoint's safetensors header (81 tensors), not from the config:
//! `fc` [4096][20480] = five taps x 4096, `candidate_selector.hidden_projection` [256][4096],
//! `{predecessor,successor}_codebook` [154880][256], `attention_conv.kernel_projection` and
//! `mlp_conv.kernel_projection` [1024][4096] = [which][tap][256 groups], and
//! `*_conv.base_kernel` [2][2][4096] = [which][tap][channel].

use tidelm_cuda::DeviceBuffer;
use tidelm_kernels::dflash::{D, KVW};

use crate::weights::Loader;

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
    pub fn load(ck: &Loader<'_>) -> Result<Self, String> {
        let one = |name: &str| -> Result<DeviceBuffer<u16>, String> {
            ck.bf16(name).map_err(|e| format!("{name}: {e}"))
        };
        // Rows of several tensors into one buffer, in the order the kernels read them.
        let stack = |parts: &[(&str, usize)]| -> Result<DeviceBuffer<u16>, String> {
            let mut host: Vec<u16> = Vec::new();
            for (name, rows) in parts {
                let b = ck.bf16(name).map_err(|e| format!("{name}: {e}"))?;
                let v = b.download().map_err(|e| format!("{name}: download: {e}"))?;
                if v.len() != rows * D {
                    return Err(format!("{name}: {} values, expected {} x {D}", v.len(), rows));
                }
                host.extend_from_slice(&v);
            }
            DeviceBuffer::from_slice(&host).map_err(|e| format!("stack: {e}"))
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

//! The DFlash2 drafter's forward, on the real checkpoint: load, one block pass, timings.
//!
//!   glm_bench_dflash --load <dir> [--rows 8] [--iters 20]
//!
//! `--load` is the drafter's own directory (`models/glm53-dflash2`, 2.2 GB bf16). The taps are a
//! deterministic pattern, not a target's capture: this bench measures the drafter's forward cost
//! and proves the five layers run to finite outputs, not acceptance. Acceptance needs the target's
//! taps and its head, which is the engine's job (GLM_DRAFTER=dflash2), not a bench's.
//!
//! What it prints: the weight bytes against the published 2.2 GB, the ring's bytes a stream, and
//! per phase -- embed (fc + hidden_norm), the five layers together, and the selector projection --
//! as the median of `--iters`. Then the block hidden's statistics: finite count, min/max, and the
//! share of exact zeros, which is how a silently dead kernel shows up here.

use std::time::Instant;

use tidelm_cuda::{DeviceBuffer, Stream};
use tidelm_glm::dflash::{Drafter, BLOCK, LAYERS};
use tidelm_glm::dflash_fwd::Run;
use tidelm_glm::weights::Loader;
use tidelm_glm::Result;
use tidelm_kernels::blas::Blas;
use tidelm_kernels::dflash as k;
use tidelm_model::Checkpoint;

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned()
}

fn main() -> Result<()> {
    let Some(dir) = arg("--load") else {
        println!("usage: glm_bench_dflash --load <drafter dir> [--rows 8] [--iters 20]");
        return Ok(());
    };
    let rows: usize = arg("--rows").and_then(|v| v.parse().ok()).unwrap_or(BLOCK);
    let iters: usize = arg("--iters").and_then(|v| v.parse().ok()).unwrap_or(20);
    if rows == 0 || rows % BLOCK != 0 {
        return Err(format!("--rows must be a multiple of {BLOCK}").into());
    }

    tidelm_cuda::set_device(0)?;
    let st = Stream::new()?;
    let ckpt = Checkpoint::open(std::path::Path::new(&dir))?;
    let loader = Loader { ck: &ckpt, rank: 0, world: 1, st: &st };
    let t0 = Instant::now();
    let dr = Drafter::load(&loader)?;
    println!("loaded in {:.1} s: {:.2} GB of weights (published 2.2 GB), {LAYERS} layers", t0.elapsed().as_secs_f32(),
             dr.nbytes() as f32 / 1e9);

    let run = Run::new(&dr, rows)?;
    println!("ring {:.1} MiB a stream ({} streams), rows {rows}", dr.ring_row as f32 * 2.0 / (1024.0 * 1024.0),
             run.streams);

    // The taps: a deterministic pattern, finite and non-constant, so a broken layer shows as NaN
    // rather than as a plausible constant.
    let mut seed = 0x2545F4914F6CDD1Du64;
    // Finite bf16 values in [-1, 1]: raw random bits are mostly NaN in bf16, which would blame the
    // kernels for the bench's own input.
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let f = ((seed >> 40) as f32 / 8388608.0) - 1.0;
        (f.to_bits() >> 16) as u16
    };
    let taps: Vec<u16> = (0..rows * 5 * k::D).map(|_| rnd()).collect();
    let cin = DeviceBuffer::from_slice(&taps)?;
    run.cin.copy_from(0, &cin, 0, taps.len(), &st)?;
    // One stream, starting at position 0 with no context: the attention sees the block's own keys.
    let starts = vec![0i64; run.streams];
    let froms = vec![0i64; run.streams];
    let bpos: Vec<i64> = (0..rows).map(|r| r as i64).collect();
    run.set_tables(&st, &starts, &froms, &bpos)?;

    let blas = Blas::new(st.raw())?;
    // Warm: first touch of 2.2 GB of weights is the weight, not the arithmetic.
    run.embed_taps(&blas, &dr)?;
    run.pass(&st, &blas, &dr)?;
    st.synchronize()?;

    let mut embed = Vec::new();
    let mut layers = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        run.embed_taps(&blas, &dr)?;
        st.synchronize()?;
        embed.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        run.pass(&st, &blas, &dr)?;
        st.synchronize()?;
        layers.push(t.elapsed().as_secs_f64() * 1e3);
    }
    st.synchronize()?;
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("embed (fc + hidden_norm): {:.3} ms", med(&mut embed));
    println!("five layers + norm + selector projection: {:.3} ms", med(&mut layers));

    let hn = run.hn.download()?;
    let g = run.g.download()?;
    let finite = hn.iter().filter(|&&x| hf(x).is_finite()).count();
    let zeros = hn.iter().filter(|&&x| x == 0).count();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &hn {
        let f = hf(x);
        if f.is_finite() {
            lo = lo.min(f);
            hi = hi.max(f);
        }
    }
    println!("hn: {finite}/{} finite, {zeros} exact zeros, range [{lo:.4}, {hi:.4}]", hn.len());
    let gfin = g.iter().filter(|&&x| hf(x).is_finite()).count();
    println!("g (selector projection): {gfin}/{} finite", g.len());
    if finite != hn.len() || gfin != g.len() {
        return Err("the drafter's forward produced non-finite values".into());
    }
    Ok(())
}

/// bf16 bits to f32, for the statistics above (the crate's own conversion, not a second one).
fn hf(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

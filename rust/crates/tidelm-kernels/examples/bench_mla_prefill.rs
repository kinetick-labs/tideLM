//! GLM-5.3-Flash sparse-MLA prefill (glm_mla_nope, one split) at one 4096-row chunk: T rows x 64 heads
//! over K = 2048 sorted picks of a 32K-row FP8 latent cache; ms a call (best of 5). --probe N sets
//! glm.cu's c_mla_probe (1 no key loads, 2 no score mma, 4 no PV mma, 16 no FP8 widening). The
//! opt-in 64-head block: TIDELM_MLA_FP8K=1 in the environment.
//!
//!   cargo run --release -p tidelm-kernels --example bench_mla_prefill [--rows 4096] [--probe 0,1,2,4,16]

use std::time::Instant;

use tidelm_cuda::{check, set_device, DeviceBuffer, Stream};
use tidelm_kernels::glm::{glm_mla_nope, glm_set_mla_probe};

fn arg(name: &str, d: &str) -> String {
    std::env::args().skip_while(|a| a != name).nth(1).unwrap_or_else(|| d.into())
}

fn main() {
    set_device(0).unwrap();
    let st = Stream::new().unwrap();
    let (cache, heads, k) = (32768usize, 64usize, 2048usize);
    let t: usize = arg("--rows", "4096").parse().unwrap();
    let probes: Vec<i32> = arg("--probe", "0").split(',').map(|v| v.trim().parse().unwrap()).collect();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let cq: Vec<u8> = (0..cache * 512).map(|_| (rnd() & 0x7f) as u8).collect();
    let cs: Vec<f32> = (0..cache * 4).map(|_| 0.01 + (rnd() % 100) as f32 * 1e-4).collect();
    let q: Vec<u16> = (0..t * heads * 512).map(|_| 0x3c00 | (rnd() & 0xff) as u16).collect();
    let mut idx = vec![0i32; t * k];
    for r in 0..t {
        let mut v: Vec<i32> = (0..k).map(|_| (rnd() % cache as u64) as i32).collect();
        v.sort();
        idx[r * k..(r + 1) * k].copy_from_slice(&v);
    }
    let (d_cq, d_cs, d_q, d_idx) = (DeviceBuffer::from_slice(&cq).unwrap(), DeviceBuffer::from_slice(&cs).unwrap(),
                                   DeviceBuffer::from_slice(&q).unwrap(), DeviceBuffer::from_slice(&idx).unwrap());
    let d_b = DeviceBuffer::from_slice(&vec![cache as i32 - 1; t]).unwrap();
    let pm = DeviceBuffer::<f32>::zeroed(t * heads).unwrap();
    let pl = DeviceBuffer::<f32>::zeroed(t * heads).unwrap();
    let pacc = DeviceBuffer::<f32>::zeroed(t * heads * 512).unwrap();
    let out = DeviceBuffer::<u16>::zeroed(t * heads * 512).unwrap();
    for &probe in &probes {
        unsafe { glm_set_mla_probe(probe); }
        let mut best = f64::INFINITY;
        for _ in 0..6 {
            let t0 = Instant::now();
            check(unsafe {
                glm_mla_nope(st.raw(), 2, d_q.raw(), 0, (heads * 512) as i64, 512, heads as i32, d_cq.raw(), d_cs.raw(),
                             std::ptr::null(), 0, d_idx.raw(), k as i64, k as i32, 1, d_b.raw(), t as i32, heads as i32, 0.0442, 1,
                             k as i32, pm.raw(), pl.raw(), pacc.raw(), out.raw(), (heads * 512) as i64, 512, std::ptr::null())
            }, "mla").unwrap();
            st.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!("rows {t} probe {probe:2}: {best:.2} ms");
    }
    if std::env::var("DUMP").is_ok() {
        let o = out.download().unwrap();
        let s = o.iter().fold(0xcbf29ce484222325u64, |h, &v| (h ^ v as u64).wrapping_mul(0x100000001b3));
        println!("out checksum {s}");
    }
}

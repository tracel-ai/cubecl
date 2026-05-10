//! Microbench comparing pageable vs pinned host->device upload bandwidth.
//!
//! Run with:
//!   cargo run --example upload_bench --release --features cuda

use cubecl::prelude::*;

const SIZES_MB: &[usize] = &[4, 48, 192];
const ITERS: usize = 5;
const WARMUP_ITERS: usize = 2;

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    println!(
        "{:>10} | {:>14} | {:>14} | {:>10}",
        "size (MB)", "create_slice", "create_pinned", "speedup"
    );
    println!("{}", "-".repeat(60));

    for &mb in SIZES_MB {
        let size = mb * 1024 * 1024;
        let buf = vec![0xABu8; size];

        // Warmup both paths.
        for _ in 0..WARMUP_ITERS {
            let h = client.create_from_slice(&buf);
            drop(h);
            let h = client.create_from_slice_pinned(&buf);
            drop(h);
        }
        cubecl::future::block_on(client.sync()).unwrap();

        // Benchmark create_from_slice (current default path).
        let mut total_pageable_ns: u128 = 0;
        for _ in 0..ITERS {
            cubecl::future::block_on(client.sync()).unwrap();
            let t0 = std::time::Instant::now();
            let h = client.create_from_slice(&buf);
            cubecl::future::block_on(client.sync()).unwrap();
            total_pageable_ns += t0.elapsed().as_nanos();
            drop(h);
        }
        let avg_pageable_ms = (total_pageable_ns as f64 / ITERS as f64) / 1e6;
        let pageable_gbs = (size as f64 / 1e9) / (avg_pageable_ms / 1e3);

        // Benchmark create_from_slice_pinned (new fast path).
        let mut total_pinned_ns: u128 = 0;
        for _ in 0..ITERS {
            cubecl::future::block_on(client.sync()).unwrap();
            let t0 = std::time::Instant::now();
            let h = client.create_from_slice_pinned(&buf);
            cubecl::future::block_on(client.sync()).unwrap();
            total_pinned_ns += t0.elapsed().as_nanos();
            drop(h);
        }
        let avg_pinned_ms = (total_pinned_ns as f64 / ITERS as f64) / 1e6;
        let pinned_gbs = (size as f64 / 1e9) / (avg_pinned_ms / 1e3);

        let speedup = avg_pageable_ms / avg_pinned_ms;

        println!(
            "{:>10} | {:>8.2} GB/s | {:>8.2} GB/s | {:>9.2}x",
            mb, pageable_gbs, pinned_gbs, speedup
        );
    }
}

//! Cost/benefit benchmark for `QuotientRangePass`, per the review of PR #1676:
//! the pass replaces the filter loop's hole iterations with a recovery chain
//! that costs ops per surviving tap, so it only pays off when the holes are
//! numerous enough. This sweeps the inner loop's trip count (`kernel *
//! dilation`, of which `kernel` are useful) and times the same filter loop
//! with the pass on and off, plus the hand-written kernel-position loop as
//! the ideal quotient-indexed reference.
//!
//! Run twice in release, once per mode, and merge the filter columns. The
//! pass is disabled by default; opt in with `CUBECL_ENABLE_QUOTIENT_RANGE=1`.
//! The harness verifies the requested mode so the A/B data cannot be silently doubled.
//! The `std` feature is what makes the switch readable (`cubecl-opt` is a
//! `default-features = false` dependency here), and the run asserts it.
//!
//! ```bash
//! cargo run --release -p cubecl-wgpu --features std --example quotient_range_bench
//! CUBECL_ENABLE_QUOTIENT_RANGE=1 cargo run --release -p cubecl-wgpu --features std --example quotient_range_bench
//! ```
//! On macOS, use `--features msl,std` for the native Metal compiler.
//!
//! Each timed launch sweeps `in_c * len` = 524288 outer positions, the long
//! outer loop the review asked for to make minor variations timeable. The
//! server disables persistent compilation caching for explicit experiments,
//! keeping their artifacts out of the default cache.
//! Numbers are per device and per driver and do not transfer between them, so
//! quote the adapter alongside them; the run prints it.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::dilated_conv_transpose::{
    Problem, launch_filter, launch_kernel_pos, prepare_buffers,
};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;
use std::time::{Duration, Instant};

type R = WgpuRuntime;

const IN_C: u32 = 128;
const LENGTH: u32 = 4096;
const KERNELS: [u32; 4] = [3, 5, 9, 17];
const DILATIONS: [u32; 6] = [1, 2, 3, 4, 8, 16];

/// Alternate measurement order to reduce systematic clock/temperature drift.
/// Times include launch + sync; use a release build to limit host overhead.
fn time_launches(
    client: &Client,
    mut filter: impl FnMut(&Client),
    mut kpos: impl FnMut(&Client),
) -> (f64, f64) {
    let measure = |launch: &mut dyn FnMut(&Client)| {
        let start = Instant::now();
        launch(client);
        cubecl_core::future::block_on(client.sync()).unwrap();
        start.elapsed()
    };
    for _ in 0..3 {
        measure(&mut filter);
        measure(&mut kpos);
    }
    let mut filter_samples = Vec::with_capacity(10);
    let mut kpos_samples = Vec::with_capacity(10);
    for sample in 0..10 {
        if sample % 2 == 0 {
            filter_samples.push(measure(&mut filter));
            kpos_samples.push(measure(&mut kpos));
        } else {
            kpos_samples.push(measure(&mut kpos));
            filter_samples.push(measure(&mut filter));
        }
    }
    (median_ms(filter_samples), median_ms(kpos_samples))
}

fn median_ms(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1_000.0
}

fn main() {
    let enabled = cubecl_opt::passes::quotient_range::enabled();
    let mode = if enabled {
        "pass ON (experimental)"
    } else {
        "pass OFF (baseline, default)"
    };
    assert_eq!(
        enabled,
        std::env::var("CUBECL_ENABLE_QUOTIENT_RANGE").as_deref() == Ok("1")
            && std::env::var_os("CUBECL_DISABLE_QUOTIENT_RANGE").is_none(),
        "the pass switch is inert (cubecl-opt built without std?); \
         rebuild with `--features std` or the A/B data is meaningless"
    );
    let client = R::client(&Default::default());

    println!(
        "adapter: {} ({})",
        client.properties().identity.name,
        client.name()
    );
    println!("quotient-range cost/benefit — mode: {mode}");
    println!(
        "{:>3} {:>3} {:>7} {:>7} {:>12} {:>12} {:>10}",
        "k", "d", "N=k*d", "useful", "filter ms", "kpos ms", "filt/kpos"
    );

    for &k in &KERNELS {
        for &d in &DILATIONS {
            let problem = Problem::same_size(IN_C, LENGTH, k, d);
            assert_eq!(
                problem.out_len(),
                LENGTH,
                "same-size padding must keep out_len fixed"
            );
            let (input, weight, output) = prepare_buffers(&client, problem);

            // The two loop forms must agree before either is timed; this also
            // pins the launches against dead-code elimination.
            let filter_out = launch_filter(&client, &input, &weight, output.clone(), problem);
            // Read before the second launch overwrites the shared output buffer.
            let filter_out = f32::from_bytes(&client.read_one_unchecked(filter_out)).to_vec();
            let kpos_out = launch_kernel_pos(&client, &input, &weight, output.clone(), problem);
            let kpos_out = f32::from_bytes(&client.read_one_unchecked(kpos_out)).to_vec();
            assert_eq!(filter_out.len(), kpos_out.len());
            for (i, (a, b)) in filter_out.iter().zip(kpos_out.iter()).enumerate() {
                let tol = 1e-4 * b.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "k={k} d={d} index {i}: filter={a} kpos={b}"
                );
            }

            let (filter_ms, kpos_ms) = time_launches(
                &client,
                |client| {
                    launch_filter(client, &input, &weight, output.clone(), problem);
                },
                |client| {
                    launch_kernel_pos(client, &input, &weight, output.clone(), problem);
                },
            );
            println!(
                "{k:>3} {d:>3} {:>7} {:>7} {filter_ms:>12.3} {kpos_ms:>12.3} {:>9.2}x",
                k * d,
                k,
                filter_ms / kpos_ms
            );
        }
    }
    println!(
        "mode: {mode} — compare the filter columns across the two runs; kpos is the shared reference"
    );
}

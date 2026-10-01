//! Compare the optimized input-window loop with a hand-written kernel-position loop.
//!
//! QuotientRangePass is registered by default and rewrites windows when its
//! safety and profitability checks succeed. This benchmark sweeps kernel size
//! and dilation, checking correctness before timing both loop implementations.
//!
//! ```bash
//! cargo run --release -p cubecl-wgpu --features std --example quotient_range_bench
//! ```
//! On macOS, use `--features msl,std` for the native Metal compiler.
//!
//! Sweep 1/8/128/512 channels at length 4096, plus clipped windows. Channels
//! are serial outer iterations per output, not additional GPU threads.
//! To measure the pass's impact, run the same harness in
//! an isolated checkout with only the QuotientRangePass registrations removed.
//! Use the same release profile, device and workload for both builds. The kernel-position loop is
//! a reference implementation, not a measurement of the pass being disabled.
//! Numbers are device- and driver-specific; the run prints the adapter.

use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::dilated_conv_transpose::{
    Problem, launch_filter, launch_kernel_pos, prepare_buffers,
};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;
use std::io::Write;
use std::time::{Duration, Instant};

type R = WgpuRuntime;

const CHANNELS: [u32; 4] = [1, 8, 128, 512];
const LENGTH: u32 = 4096;
const KERNELS: [u32; 5] = [1, 3, 5, 9, 17];
const DILATIONS: [u32; 6] = [1, 2, 3, 4, 8, 16];

/// Alternate measurement order to reduce systematic clock/temperature drift.
/// Times include launch + sync; use a release build to limit host overhead.
fn time_launches(
    client: &Client,
    samples: usize,
    mut filter: impl FnMut(&Client),
    mut kpos: impl FnMut(&Client),
) -> (Samples, Samples) {
    let measure = |launch: &mut dyn FnMut(&Client)| {
        let start = Instant::now();
        launch(client);
        cubecl_core::future::block_on(client.sync()).unwrap();
        start.elapsed()
    };
    for _ in 0..10 {
        measure(&mut filter);
        measure(&mut kpos);
    }
    let mut filter_samples = Vec::with_capacity(samples);
    let mut kpos_samples = Vec::with_capacity(samples);
    for sample in 0..samples {
        if sample % 2 == 0 {
            filter_samples.push(measure(&mut filter));
            kpos_samples.push(measure(&mut kpos));
        } else {
            kpos_samples.push(measure(&mut kpos));
            filter_samples.push(measure(&mut filter));
        }
    }
    (Samples::new(filter_samples), Samples::new(kpos_samples))
}

struct Samples {
    min: f64,
    median: f64,
    max: f64,
    raw: Vec<Duration>,
}

impl Samples {
    fn new(raw: Vec<Duration>) -> Self {
        let mut times = raw.clone();
        times.sort_unstable();
        let middle = times.len() / 2;
        Self {
            min: times[0].as_secs_f64() * 1000.0,
            median: (times[middle].as_secs_f64() + times[(times.len() - 1) / 2].as_secs_f64())
                * 500.0,
            max: times[times.len() - 1].as_secs_f64() * 1000.0,
            raw,
        }
    }
}

fn main() {
    let mut samples = 100usize;
    let mut channels = CHANNELS.to_vec();
    let mut raw_output = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--samples" => {
                samples = args
                    .next()
                    .expect("--samples needs a count")
                    .parse()
                    .unwrap();
                assert!(samples > 0);
            }
            "--channels" => {
                channels = args
                    .next()
                    .expect("--channels needs a comma-separated list")
                    .split(',')
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect();
                assert!(channels.iter().all(|&c| c > 0));
            }
            "--raw" => {
                let file =
                    std::fs::File::create(args.next().expect("--raw needs a CSV path")).unwrap();
                raw_output = Some(std::io::BufWriter::new(file));
            }
            _ => {
                panic!("unknown argument {arg}; use --samples N --channels 1,8,128,512 --raw PATH")
            }
        }
    }
    if let Some(raw) = &mut raw_output {
        writeln!(
            raw,
            "channels,length,kernel,dilation,stride,sample,filter_ms,kpos_ms"
        )
        .unwrap();
    }
    let client = R::client(&Default::default());

    println!(
        "adapter: {} ({})",
        client.properties().identity.name,
        client.name()
    );
    println!(
        "channels,length,kernel,dilation,stride,filter_min_ms,filter_median_ms,filter_max_ms,kpos_median_ms"
    );
    let problems = channels.into_iter().flat_map(|in_c| {
        let interior = KERNELS.into_iter().flat_map(move |kernel| {
            DILATIONS
                .into_iter()
                .map(move |dilation| Problem::same_size(in_c, LENGTH, kernel, dilation))
        });
        let boundaries = [3, 8, 16].into_iter().flat_map(move |dilation| {
            let clipped = Problem::same_size(in_c, 16, 5, dilation);
            let mut strided = Problem::same_size(in_c, LENGTH, 5, dilation);
            strided.stride = 2;
            [clipped, strided]
        });
        interior.chain(boundaries)
    });
    for problem in problems {
        let (k, d) = (problem.kernel, problem.dilation);
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
            samples,
            |client| {
                launch_filter(client, &input, &weight, output.clone(), problem);
            },
            |client| {
                launch_kernel_pos(client, &input, &weight, output.clone(), problem);
            },
        );
        println!(
            "{},{},{k},{d},{},{:.6},{:.6},{:.6},{:.6}",
            problem.in_c,
            problem.in_len,
            problem.stride,
            filter_ms.min,
            filter_ms.median,
            filter_ms.max,
            kpos_ms.median,
        );
        if let Some(raw) = &mut raw_output {
            for (sample, (filter, kpos)) in filter_ms.raw.iter().zip(&kpos_ms.raw).enumerate() {
                writeln!(
                    raw,
                    "{},{},{k},{d},{},{sample},{:.6},{:.6}",
                    problem.in_c,
                    problem.in_len,
                    problem.stride,
                    filter.as_secs_f64() * 1000.0,
                    kpos.as_secs_f64() * 1000.0
                )
                .unwrap();
            }
            raw.flush().unwrap();
        }
    }
}

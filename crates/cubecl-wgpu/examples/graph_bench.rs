//! What graph replay is worth on wgpu: per-launch cost of the normal launch
//! path against replaying a captured graph, at sizes where launch overhead
//! dominates and at sizes where it cannot possibly matter.
//!
//! Run with `cargo run --release -p cubecl-wgpu --example graph_bench`. Numbers
//! are per device and per driver and do not transfer between them, so quote the
//! adapter alongside them; the run prints it.
//!
//! **The shape.** One *pass* is a chain of dependent kernels ping-ponging
//! between two buffers — the decode-loop shape, where each launch's cost is
//! exposed rather than hidden behind a queue of independent work. A pass is
//! issued and then synced, as a decode step is.
//!
//! **How it measures.** The two paths are interleaved round by round, never all
//! of one then all of the other, so a thermal or clock drift over the run lands
//! on both instead of on whichever went second. Each round reports:
//!
//! - *enqueue* — wall time for the calling thread to hand the pass over, before
//!   any sync. Read this one carefully and never as a speedup: the normal path
//!   posts one message per launch, while a replay posts exactly one whatever
//!   its length, and the re-encoding it triggers happens on the server thread
//!   *after* the call returns. So this column says how much work leaves the
//!   caller's thread, not how much work disappears.
//! - *e2e* — enqueue plus the sync that waits for the GPU, which is where the
//!   re-encoding is actually paid for. Both paths run identical kernels, so this
//!   is the column a win has to survive in to be real.
//! - *share* — enqueue as a fraction of e2e on the normal path. It says whether
//!   launch overhead is worth attacking at that size at all; a large speedup on
//!   a small share is a large speedup on nothing.
//!
//! **The control.** The `1m` configs are GPU-bound: a pass takes long enough
//! that enqueue is a rounding error, so replay must show ~1.0x e2e there. A
//! "win" that also appears on those rows is measuring the harness, not the
//! change.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_wgpu::WgpuRuntime;
use std::time::{Duration, Instant};

#[cube(launch)]
fn add_one_tensor(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// One point on the sweep: how big each kernel is, how many of them chain into
/// a pass, and how many rounds to average over.
struct Config {
    name: &'static str,
    elems: usize,
    kernels: usize,
    rounds: usize,
}

/// Small kernels first (launch-bound, where replay should win), then the
/// GPU-bound control at the bottom.
const CONFIGS: &[Config] = &[
    Config {
        name: "tiny x150",
        elems: 64,
        kernels: 150,
        rounds: 100,
    },
    Config {
        name: "tiny x1000",
        elems: 64,
        kernels: 1000,
        rounds: 50,
    },
    Config {
        name: "tiny x5000",
        elems: 64,
        kernels: 5000,
        rounds: 10,
    },
    Config {
        name: "64k x1000",
        elems: 64 * 1024,
        kernels: 1000,
        rounds: 20,
    },
    Config {
        name: "256k x1000",
        elems: 256 * 1024,
        kernels: 1000,
        rounds: 10,
    },
    Config {
        name: "1m x500",
        elems: 1024 * 1024,
        kernels: 500,
        rounds: 10,
    },
    Config {
        name: "1m x2000",
        elems: 1024 * 1024,
        kernels: 2000,
        rounds: 5,
    },
];

const CUBE_DIM: u32 = 256;

fn main() {
    let client = WgpuRuntime::client(&Default::default());

    println!("adapter: {}", WgpuRuntime::name(&client));
    println!(
        "{:>11} | {:>19} | {:>24} | {:>6} | {:>8}",
        "config", "enqueue µs/pass", "e2e µs/kernel", "share", "capture"
    );
    println!(
        "{:>11} | {:>9} {:>9} | {:>9} {:>9} {:>4}| {:>6} | {:>8}",
        "", "normal", "replay", "normal", "replay", "x", "", ""
    );

    for config in CONFIGS {
        bench(&client, config);
    }
}

fn bench(client: &ComputeClient<WgpuRuntime>, config: &Config) {
    let a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; config.elems]));
    let b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; config.elems]));

    // Warm the compilation cache, the pools and the info cache, so neither path
    // pays a first-run cost the other does not.
    run_pass(client, &a, &b, config);
    sync(client, &a);

    // Capture before timing anything: the recorded pass is the same sequence the
    // normal path runs, and `graph_prepare` needs its own warmup run.
    client.graph_prepare().expect("graph_prepare");
    run_pass(client, &a, &b, config);
    sync(client, &a);
    client.start_capture().expect("start_capture");
    run_pass(client, &a, &b, config);
    let capture_start = Instant::now();
    let graph = client.stop_capture().expect("stop_capture");
    let capture = capture_start.elapsed();
    unsafe { graph.replay() };
    sync(client, &a);

    let mut normal = Measure::default();
    let mut replay = Measure::default();
    for _ in 0..config.rounds {
        normal.round(|| run_pass(client, &a, &b, config), || sync(client, &a));
        replay.round(|| unsafe { graph.replay() }, || sync(client, &a));
    }

    let launches = (config.rounds * config.kernels) as f64;
    let passes = config.rounds as f64;
    let share = normal.issue.as_secs_f64() / normal.total.as_secs_f64();
    println!(
        "{:>11} | {:>9.2} {:>9.2} | {:>9.2} {:>9.2} {:>3.1}x| {:>5.1}% | {:>8.2?}",
        config.name,
        per(normal.issue, passes),
        per(replay.issue, passes),
        per(normal.total, launches),
        per(replay.total, launches),
        ratio(normal.total, replay.total),
        share * 100.0,
        capture,
    );
}

/// Accumulated time for one path, split at the sync so the caller-side cost
/// stays separable from the GPU work behind it.
#[derive(Default)]
struct Measure {
    /// Time in the enqueue call itself, before any wait.
    issue: Duration,
    /// `issue` plus the sync that follows it.
    total: Duration,
}

impl Measure {
    fn round(&mut self, issue: impl FnOnce(), sync: impl FnOnce()) {
        let start = Instant::now();
        issue();
        self.issue += start.elapsed();
        sync();
        self.total += start.elapsed();
    }
}

fn run_pass(client: &ComputeClient<WgpuRuntime>, a: &Handle, b: &Handle, config: &Config) {
    let cubes = config.elems.div_ceil(CUBE_DIM as usize) as u32;
    for i in 0..config.kernels {
        let (src, dst) = if i % 2 == 0 { (a, b) } else { (b, a) };
        add_one_tensor::launch(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            // SAFETY: contiguous rank-1 buffers of exactly `elems` f32
            // elements, matching the declared shape and strides.
            unsafe { TensorArg::from_raw_parts(src.clone(), [1].into(), [config.elems].into()) },
            unsafe { TensorArg::from_raw_parts(dst.clone(), [1].into(), [config.elems].into()) },
        );
    }
}

fn sync(client: &ComputeClient<WgpuRuntime>, handle: &Handle) {
    let _ = client.read_one(handle.clone()).unwrap();
}

fn per(elapsed: Duration, count: f64) -> f64 {
    elapsed.as_secs_f64() * 1e6 / count
}

fn ratio(before: Duration, after: Duration) -> f64 {
    before.as_secs_f64() / after.as_secs_f64()
}

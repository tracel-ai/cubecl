//! What the process does with the work it is asked to run, shown on a small
//! autotuned workload: a vector scaled by two factors at three sizes, each
//! launch picking between a scalar and a vectorized kernel by measuring them.
//!
//! - **A direct run**, for comparison: the workload executes from a cold
//!   start, compiling and tuning as it reaches each kernel and key, then once
//!   more warmed up.
//! - **A build** runs the workload twice. Under
//!   [`ProcessMode::CompileOnly`] every launch and every tune only queues its
//!   kernels; under [`ProcessMode::CompileAndAutotune`] the queue compiles in
//!   one batch and the tunes measure. Neither runs the workload itself. A
//!   second thread reads the build's counts while it runs.
//! - **A warmed-up run** executes for real under [`ProcessMode::Execute`],
//!   counted all the same: it compiles and tunes nothing, since the build
//!   left both done.
//! - **A measurement** runs while the process discards everything else: a
//!   [`StreamModeOverride`] puts one client's stream back in
//!   [`StreamMode::Execute`].
//!
//! Each prints how long it took. Both start cold: the example keeps its
//! environments in a directory of its own, emptied for every run, and the two
//! scale by different factors, so neither reuses a kernel or a tune of the
//! other's — a runtime that does not store its kernels keeps them across
//! environments.

use cubecl::{
    Device,
    client::Client,
    execution::{
        ExecutionStatistics, ProcessMode, ProcessModeOverride, StatisticsCollector,
        StatisticsReader, StreamMode, StreamModeOverride,
    },
    future::block_on,
    prelude::*,
    server::Handle,
    tune::{CloneInputGenerator, LocalTuner, Tunable, TunableSet, local_tuner},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Every element of `input` times `factor`, into `output`.
#[cube(launch_unchecked)]
fn scale<F: Float, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    #[comptime] factor: u32,
) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * Vector::new(F::new(comptime!(factor as f32)));
    }
}

/// The direct run's factors: each is a kernel of its own, since it is known
/// at compile time.
const DIRECT_FACTORS: [u32; 2] = [2, 3];
/// The build's factors: as many as the direct run's, and none of them.
const BUILD_FACTORS: [u32; 2] = [5, 7];
/// The workload's sizes: each is an autotune key of its own.
const LENGTHS: [usize; 3] = [1 << 12, 1 << 16, 1 << 20];
/// How many elements one unit scales per launch row.
const UNITS: u32 = 256;

static TUNER: LocalTuner<String, String> = local_tuner!("scale");

/// Run the workload directly, then build it and run it warmed up, timing
/// each, and measure beside a process that discards its launches.
pub fn run(device: &Device) {
    let client = device.client();
    // After the client: bringing it up applies cubecl's configuration, which
    // sets the root of its own.
    let root = std::env::temp_dir().join(format!("cubecl-execution-{}", std::process::id()));
    cubecl::environment::set_root(&root);
    println!("Running on {}", client.name());

    cubecl::environment::activate("direct");
    let cold = timed(&client, || workload(&client, DIRECT_FACTORS));
    let warmed = timed(&client, || workload(&client, DIRECT_FACTORS));
    println!(
        "Direct run: {} cold, compiling and tuning as it goes; {} warmed up",
        Millis(cold),
        Millis(warmed)
    );

    cubecl::environment::activate("build");
    let build = StatisticsCollector::new();
    let mut passes = Vec::new();
    let built = watching(build.reader(), || {
        for mode in [ProcessMode::CompileOnly, ProcessMode::CompileAndAutotune] {
            let pass = ProcessModeOverride::new(mode, &build);
            // Drained under the override, which `timed` does: a launch
            // issued after it closes runs for real.
            passes.push((mode, timed(&client, || workload(&client, BUILD_FACTORS))));
            core::mem::drop(pass);
        }
    });
    for (mode, took) in &passes {
        println!("{mode:?} pass: {}", Millis(*took));
    }
    println!("Built: {}", Summary(built));

    let warm = StatisticsCollector::new();
    let counted = ProcessModeOverride::new(ProcessMode::Execute, &warm);
    let executed = timed(&client, || workload(&client, BUILD_FACTORS));
    core::mem::drop(counted);
    println!(
        "Built run: {} executing, having compiled nothing and tuned nothing: {}",
        Millis(executed),
        Summary(warm.statistics())
    );
    let building: Duration = passes.iter().map(|(_, took)| *took).sum();
    println!(
        "Build and run: {}, against {} run directly",
        Millis(building + executed),
        Millis(cold)
    );

    measurement_beside_discarded_work(&client);
    let _ = std::fs::remove_dir_all(root);
}

/// How long `work` takes, its launches drained.
fn timed(client: &Client, work: impl FnOnce()) -> Duration {
    let started = Instant::now();
    work();
    block_on(client.sync()).expect("the device is up");
    started.elapsed()
}

/// The workload: every factor at every length, each launch autotuned.
fn workload(client: &Client, factors: [u32; 2]) {
    for factor in factors {
        let tuned = format!("scale-x{factor}");
        let set = TUNER.init(&tuned, {
            let client = client.clone();
            move || scale_candidates(&client, factor)
        });
        for length in LENGTHS {
            let input = client.create_from_slice(f32::as_bytes(&vec![1.0; length]));
            let output = client.empty(length * size_of::<f32>());
            TUNER.execute(&tuned, client, set.clone(), vec![input, output]);
        }
    }
}

/// The candidates autotune picks between for one `factor`: one element per
/// unit, or four.
fn scale_candidates(client: &Client, factor: u32) -> TunableSet<String, Vec<Handle>, ()> {
    let (scalar, vectorized) = (client.clone(), client.clone());
    TunableSet::new(
        move |inputs: &Vec<Handle>| {
            format!(
                "scale-x{factor}-{}",
                inputs[0].size() as usize / size_of::<f32>()
            )
        },
        CloneInputGenerator,
    )
    .with(Tunable::new("scalar", move |inputs: Vec<Handle>| {
        launch_scale(&scalar, &inputs, VectorSize(1), factor);
        Ok::<(), String>(())
    }))
    .with(Tunable::new("vectorized", move |inputs: Vec<Handle>| {
        launch_scale(&vectorized, &inputs, VectorSize(4), factor);
        Ok::<(), String>(())
    }))
}

/// How many elements a [`scale`] unit reads at once.
#[derive(Clone, Copy)]
struct VectorSize(usize);

/// Launch [`scale`] over `inputs`, an input and an output of one length.
fn launch_scale(client: &Client, inputs: &[Handle], vector: VectorSize, factor: u32) {
    let length = inputs[0].size() as usize / size_of::<f32>();
    let rows = (length / vector.0).div_ceil(UNITS as usize) as u32;
    unsafe {
        scale::launch_unchecked::<f32>(
            client,
            CubeCount::Static(rows, 1, 1),
            CubeDim::new_1d(UNITS),
            vector.0,
            BufferArg::from_raw_parts(inputs[0].clone(), length),
            BufferArg::from_raw_parts(inputs[1].clone(), length),
            factor,
        )
    };
}

/// Run `build` while another thread prints `reader`'s counts as they move,
/// and return them as they ended.
fn watching(reader: StatisticsReader, build: impl FnOnce()) -> ExecutionStatistics {
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut shown = ExecutionStatistics::default();
            while !done.load(Ordering::Acquire) {
                let now = reader.statistics();
                if now != shown {
                    println!("  building: {}", Summary(now));
                    shown = now;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });
        build();
        done.store(true, Ordering::Release);
    });
    reader.statistics()
}

/// Under [`ProcessMode::CompileAndAutotune`] a launch compiles and is
/// discarded, its output untouched; on a stream put back in
/// [`StreamMode::Execute`] — what autotune does for its measurements — the
/// same launch runs.
fn measurement_beside_discarded_work(client: &Client) {
    let length = 1 << 12;
    let input = client.create_from_slice(f32::as_bytes(&vec![1.0; length]));
    let output = || client.create_from_slice(f32::as_bytes(&vec![0.0; length]));
    let first = |output: &Handle| {
        let bytes = client.read_one(output.clone()).expect("the device is up");
        f32::from_bytes(&bytes)[0]
    };

    let collector = StatisticsCollector::new();
    let tune = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    let discarded = output();
    launch_scale(
        client,
        &[input.clone(), discarded.clone()],
        VectorSize(4),
        2,
    );
    println!("Discarded launch: output[0] = {}", first(&discarded));

    let executed = output();
    {
        let measuring = StreamModeOverride::new(StreamMode::Execute, client);
        launch_scale(client, &[input, executed.clone()], VectorSize(4), 2);
        core::mem::drop(measuring);
    }
    println!("Measured launch:  output[0] = {}", first(&executed));
    core::mem::drop(tune);
}

/// [`ExecutionStatistics`] in one line.
struct Summary(ExecutionStatistics);

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ExecutionStatistics {
            compilation,
            autotune,
        } = self.0;
        write!(
            f,
            "kernels {}/{} ({} compiled, {} loaded, {} failed) · tunes {}/{} ({} measured, {} failed)",
            compilation.settled(),
            compilation.registered,
            compilation.compiled,
            compilation.loaded,
            compilation.failed,
            autotune.settled(),
            autotune.registered,
            autotune.measured,
            autotune.failed,
        )
    }
}

/// A [`Duration`] in milliseconds.
struct Millis(Duration);

impl std::fmt::Display for Millis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} ms", self.0.as_secs_f64() * 1000.0)
    }
}

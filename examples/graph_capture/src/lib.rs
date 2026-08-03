//! Runnable CUDA graph capture and replay examples.
//!
//! A graph capture records a fixed-shape sequence of kernel launches once, then replays
//! the whole sequence as a single dispatch. The lifecycle is:
//!
//! ```text
//! graph_prepare()  ->  warmup runs  ->  sync  ->  start_capture()
//!                                                 ->  the work (recorded, NOT executed)
//!                                                 ->  stop_capture()  ->  Graph
//! graph.replay()   (many times, refreshing input buffers in place between replays)
//! ```
//!
//! # Why warmup is mandatory
//!
//! An allocation issued while a capture is recording is captured as a *memory node*, and
//! the driver refuses to relaunch a graph holding an allocation node it never frees: the
//! first replay succeeds and every later one fails. So the capture window must allocate
//! nothing. Two rules make that true, and both examples here follow them:
//!
//! 1. Allocate every buffer the window touches **before** `graph_prepare`.
//! 2. Run the exact closure that will be recorded a few times as warmup, so the memory
//!    pool is primed and the window reuses slices instead of growing.
//!
//! # One capture at a time
//!
//! Capture is device-global: any concurrent allocation or sync in the process aborts a
//! recording capture. Keep all work on a single client, and never run two captures
//! concurrently on one device.

use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::future;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::time::Instant;

/// Elements in every buffer. Fixed on purpose: a capture records one set of shapes and
/// replays exactly those, so nothing here may vary per step.
const N: usize = 1024;

/// Units per cube. Paired with an explicit cube count below so all `N` elements are
/// covered — do not switch to `CubeDim::new(client, N)`, which sizes to the hardware's
/// plane layout and would leave most of the buffer untouched at this `N`.
const CUBE_DIM: u32 = 256;

/// Steps per phase.
const STEPS: usize = 10;

/// Eager passes before recording. Resolves any lazy compilation and primes the memory
/// pool so the capture window has nothing left to allocate.
const WARMUP: usize = 3;

type Res = Result<(), Box<dyn core::error::Error>>;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// A step that carries state: `state` is read, updated, and written every step, and the
/// updated value is also the step's output. This is the minimal stand-in for a recurrent
/// cache — the kind of layer whose per-step buffer, if it were allocated inside the
/// capture window, would become a memory node and make the graph un-relaunchable.
#[cube(launch)]
fn accumulate(input: &[f32], state: &mut [f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let next = state[ABSOLUTE_POS] + input[ABSOLUTE_POS];
        state[ABSOLUTE_POS] = next;
        output[ABSOLUTE_POS] = next;
    }
}

/// Cubes needed to cover `N` elements at [`CUBE_DIM`] units each.
fn cube_count() -> CubeCount {
    CubeCount::Static((N as u32).div_ceil(CUBE_DIM), 1, 1)
}

/// One `add_one` launch from `src` into `dst`.
fn add_one_launch<R: Runtime>(client: &ComputeClient<R>, src: &Handle, dst: &Handle) {
    add_one::launch::<R>(
        client,
        cube_count(),
        CubeDim::new_1d(CUBE_DIM),
        // SAFETY: both handles address `N` f32 elements, allocated by the caller and
        // kept alive for the duration of the launch.
        unsafe { BufferArg::from_raw_parts(src.clone(), N) },
        unsafe { BufferArg::from_raw_parts(dst.clone(), N) },
    );
}

/// One step: four chained launches through two alternating intermediates, so each step
/// adds exactly 4.0 to every element.
fn add_one_chain<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    scratch: &[Handle; 2],
    output: &Handle,
) {
    add_one_launch(client, input, &scratch[0]);
    add_one_launch(client, &scratch[0], &scratch[1]);
    add_one_launch(client, &scratch[1], &scratch[0]);
    add_one_launch(client, &scratch[0], output);
}

/// One `accumulate` launch.
fn accumulate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    state: &Handle,
    output: &Handle,
) {
    accumulate::launch::<R>(
        client,
        cube_count(),
        CubeDim::new_1d(CUBE_DIM),
        // SAFETY: all three handles address `N` f32 elements, allocated by the caller and
        // alive for the duration of the launch.
        unsafe { BufferArg::from_raw_parts(input.clone(), N) },
        unsafe { BufferArg::from_raw_parts(state.clone(), N) },
        unsafe { BufferArg::from_raw_parts(output.clone(), N) },
    );
}

/// Read `handle` back as a `Vec<f32>`.
fn read_f32<R: Runtime>(client: &ComputeClient<R>, handle: &Handle) -> Result<Vec<f32>, String> {
    let bytes = client
        .read_one(handle.clone())
        .map_err(|e| format!("read_one failed: {e:?}"))?;
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// Overwrite `handle` with `values`.
fn write_f32<R: Runtime>(client: &ComputeClient<R>, handle: &Handle, values: &[f32]) {
    client.write(handle, Bytes::from_bytes_vec(f32::as_bytes(values).to_vec()));
}

/// Basic fixed-shape capture/replay: run a chain eagerly, then capture it once and
/// replay it, and show the two agree.
pub fn basic<R: Runtime>(device: &R::Device) -> Res {
    let client = R::client(device);
    println!("== basic graph capture on {:?} ==", R::name(&client));

    // Rule 1: allocate everything the capture window will touch, up front.
    let seed: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let input = client.create_from_slice(f32::as_bytes(&seed));
    let scratch = [
        client.empty(N * core::mem::size_of::<f32>()),
        client.empty(N * core::mem::size_of::<f32>()),
    ];
    let output = client.empty(N * core::mem::size_of::<f32>());

    // ---- Phase 1: eager ----
    // One untimed pass first: the very first launch compiles the kernel, which costs
    // hundreds of milliseconds and has nothing to do with per-step dispatch cost. Timing
    // it would overstate the replay speedup by three orders of magnitude.
    add_one_chain::<R>(&client, &input, &scratch, &output);
    read_f32::<R>(&client, &output)?;

    let started = Instant::now();
    for _ in 0..STEPS {
        add_one_chain::<R>(&client, &input, &scratch, &output);
    }
    let eager = read_f32::<R>(&client, &output)?;
    let eager_elapsed = started.elapsed();
    println!(
        "eager:    {STEPS} steps in {eager_elapsed:?}, output[0]={:.1}",
        eager[0]
    );

    // ---- Phase 2: captured ----
    client
        .graph_prepare()
        .map_err(|e| format!("graph_prepare failed: {e:?}"))?;

    // Rule 2: warm up with the exact closure that will be recorded.
    for _ in 0..WARMUP {
        add_one_chain::<R>(&client, &input, &scratch, &output);
    }
    future::block_on(client.sync()).map_err(|e| format!("warmup sync failed: {e:?}"))?;

    client
        .start_capture()
        .map_err(|e| format!("start_capture failed: {e:?}"))?;
    // Recorded, not executed.
    add_one_chain::<R>(&client, &input, &scratch, &output);
    let graph = client
        .stop_capture()
        .map_err(|e| format!("stop_capture failed: {e:?}"))?;

    let started = Instant::now();
    for _ in 0..STEPS {
        // SAFETY: every buffer the graph recorded against is still alive above, and all
        // work stays on this one client's stream, so replays order against the reads.
        unsafe { graph.replay() };
    }
    let replayed = read_f32::<R>(&client, &output)?;
    let replay_elapsed = started.elapsed();
    println!(
        "replayed: {STEPS} steps in {replay_elapsed:?}, output[0]={:.1}",
        replayed[0]
    );

    if eager != replayed {
        return Err(format!(
            "captured replay disagreed with eager: {:?} vs {:?}",
            &eager[..4],
            &replayed[..4]
        )
        .into());
    }

    // Refreshing an input in place is what makes replay useful: the graph reads the same
    // device pointer, so new bytes there change what the next replay computes.
    let bumped: Vec<f32> = seed.iter().map(|v| v + 100.0).collect();
    write_f32::<R>(&client, &input, &bumped);
    unsafe { graph.replay() };
    let after = read_f32::<R>(&client, &output)?;
    if (after[0] - (bumped[0] + 4.0)).abs() > 1e-3 {
        return Err(format!(
            "input refresh not observed: expected {:.1}, got {:.1}",
            bumped[0] + 4.0,
            after[0]
        )
        .into());
    }
    println!(
        "after refreshing the input in place, one replay gives output[0]={:.1}",
        after[0]
    );

    println!("ok: eager and replayed agree, and the refreshed input took effect");
    Ok(())
}

/// Capture a workload that carries state across steps.
///
/// The whole point: this captures cleanly, and the only reason is *where* the state
/// buffer is allocated. It is created once here, at fixed capacity, before
/// `graph_prepare` — so the recorded window only ever reuses it. Allocating it per step
/// (inside the window) would record a memory node instead, and the driver would refuse
/// to relaunch the resulting graph past the first replay. Nothing else about a
/// state-carrying workload makes it hostile to capture.
pub fn stateful<R: Runtime>(device: &R::Device) -> Res {
    let client = R::client(device);
    println!("== stateful graph capture on {:?} ==", R::name(&client));

    let ones = vec![1.0f32; N];
    let zeros = vec![0.0f32; N];

    // Allocated once, at fixed capacity, BEFORE graph_prepare. This is the rule.
    let input = client.create_from_slice(f32::as_bytes(&ones));
    let state = client.create_from_slice(f32::as_bytes(&zeros));
    let output = client.empty(N * core::mem::size_of::<f32>());

    // ---- Phase 1: eager ----
    // One untimed pass first: the very first launch compiles the kernel, which costs
    // hundreds of milliseconds and has nothing to do with per-step dispatch cost. It also
    // advances the accumulator, so reset the state afterwards.
    accumulate_launch::<R>(&client, &input, &state, &output);
    read_f32::<R>(&client, &output)?;
    write_f32::<R>(&client, &state, &zeros);

    let started = Instant::now();
    for _ in 0..STEPS {
        accumulate_launch::<R>(&client, &input, &state, &output);
    }
    let eager = read_f32::<R>(&client, &output)?;
    let eager_elapsed = started.elapsed();
    println!(
        "eager:    {STEPS} steps in {eager_elapsed:?}, state[0]={:.1}",
        eager[0]
    );

    // ---- Phase 2: captured ----
    // Reset the state so the captured phase starts from the same place as the eager one.
    write_f32::<R>(&client, &state, &zeros);

    client
        .graph_prepare()
        .map_err(|e| format!("graph_prepare failed: {e:?}"))?;

    // Warmup mutates state, so reset again afterwards; what matters is that warmup runs
    // the identical launch, priming the pool for the window.
    for _ in 0..WARMUP {
        accumulate_launch::<R>(&client, &input, &state, &output);
    }
    future::block_on(client.sync()).map_err(|e| format!("warmup sync failed: {e:?}"))?;
    write_f32::<R>(&client, &state, &zeros);

    client
        .start_capture()
        .map_err(|e| format!("start_capture failed: {e:?}"))?;
    // Recorded, not executed — so it does NOT advance the state.
    accumulate_launch::<R>(&client, &input, &state, &output);
    let graph = client
        .stop_capture()
        .map_err(|e| format!("stop_capture failed: {e:?}"))?;

    let started = Instant::now();
    for _ in 0..STEPS {
        // SAFETY: input, state and output are all still alive above, and every replay and
        // read stays on this one client's stream.
        unsafe { graph.replay() };
    }
    let replayed = read_f32::<R>(&client, &output)?;
    let replay_elapsed = started.elapsed();
    println!(
        "replayed: {STEPS} steps in {replay_elapsed:?}, state[0]={:.1}",
        replayed[0]
    );

    if eager != replayed {
        return Err(format!(
            "captured replay disagreed with eager: {:?} vs {:?}",
            &eager[..4],
            &replayed[..4]
        )
        .into());
    }

    println!("ok: {STEPS} replays advanced the carried state exactly as {STEPS} eager steps did");
    Ok(())
}

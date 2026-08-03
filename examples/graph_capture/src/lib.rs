//! Runnable graph capture and replay examples.
//!
//! Everything here is generic over [`Runtime`], and runs on any backend implementing
//! graph capture — CUDA and HIP/ROCm today. Pick one with `--features cuda` or
//! `--features hip`.
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

/// Binds the runtime selected by the enabled cargo feature to a type alias and runs `$body`.
///
/// Keeps backend selection in one place so binaries don't each repeat the `cfg` block:
/// `dispatch!(R => graph_capture::basic::<R>(&Default::default()))`. Only backends that
/// implement graph capture are listed.
#[macro_export]
macro_rules! dispatch {
    ($runtime:ident => $body:expr) => {{
        #[cfg(feature = "cuda")]
        {
            type $runtime = cubecl::cuda::CudaRuntime;
            $body?;
        }
        #[cfg(feature = "hip")]
        {
            type $runtime = cubecl::hip::HipRuntime;
            $body?;
        }
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        println!("enable a graph-capable backend: --features cuda or --features hip");
    }};
}

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

/// Layers per step. A decode step runs the same small kernel stack once per layer, so this
/// is what sets how many launches a single replay collapses — and therefore whether
/// capture is worth anything at all.
const LAYERS: usize = 8;

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

#[cube(launch)]
fn mul_two(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

#[cube(launch)]
fn half(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 0.5;
    }
}

/// A layer that carries state: `state` is read, updated, and written every step, and the
/// updated value is also the layer's output. This is the minimal stand-in for a recurrent
/// cache — the kind of per-layer buffer whose allocation, if it happened inside the
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

macro_rules! unary_launch {
    ($name:ident, $kernel:ident) => {
        /// One `$kernel` launch from `src` into `dst`.
        fn $name<R: Runtime>(client: &ComputeClient<R>, src: &Handle, dst: &Handle) {
            $kernel::launch::<R>(
                client,
                cube_count(),
                CubeDim::new_1d(CUBE_DIM),
                // SAFETY: both handles address `N` f32 elements, allocated by the caller
                // and kept alive for the duration of the launch.
                unsafe { BufferArg::from_raw_parts(src.clone(), N) },
                unsafe { BufferArg::from_raw_parts(dst.clone(), N) },
            );
        }
    };
}

unary_launch!(add_one_launch, add_one);
unary_launch!(mul_two_launch, mul_two);
unary_launch!(half_launch, half);

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

/// A stateless decode step: [`LAYERS`] layers, each `add_one -> mul_two -> half`, which is
/// `((x + 1) * 2) / 2 = x + 1`. So a step adds exactly [`LAYERS`] to every element, while
/// issuing `3 * LAYERS` launches — the amount of dispatch a single replay collapses.
///
/// Three scratch buffers are rotated so no launch ever reads and writes the same buffer.
fn decode_step<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    scratch: &[Handle; 3],
    output: &Handle,
) {
    let mut src = input;
    for layer in 0..LAYERS {
        add_one_launch(client, src, &scratch[0]);
        mul_two_launch(client, &scratch[0], &scratch[1]);
        // The last layer lands in `output`; the others feed the next layer.
        let dst = if layer + 1 == LAYERS {
            output
        } else {
            &scratch[2]
        };
        half_launch(client, &scratch[1], dst);
        src = &scratch[2];
    }
}

/// A recurrent decode step: [`LAYERS`] layers, each transforming the running activation
/// and then folding it into that layer's own persistent state. `2 * LAYERS` launches.
///
/// Every state buffer is owned by the caller and allocated before the capture is armed —
/// that is the whole reason this captures cleanly.
fn recurrent_step<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    state: &[Handle],
    tmp: &Handle,
    act: &Handle,
    output: &Handle,
) {
    let mut src = input;
    for (layer, layer_state) in state.iter().enumerate() {
        add_one_launch(client, src, tmp);
        let dst = if layer + 1 == state.len() {
            output
        } else {
            act
        };
        accumulate_launch(client, tmp, layer_state, dst);
        src = act;
    }
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
    client.write(
        handle,
        Bytes::from_bytes_vec(f32::as_bytes(values).to_vec()),
    );
}

/// Stateless capture/replay over a layer stack: run it eagerly, then capture it once and
/// replay it, and show the two agree.
pub fn basic<R: Runtime>(device: &R::Device) -> Res {
    let client = R::client(device);
    let launches = 3 * LAYERS;
    println!("== basic graph capture on {:?} ==", R::name(&client));
    println!("   {LAYERS} layers, {launches} launches per step, {STEPS} steps");

    // Rule 1: allocate everything the capture window will touch, up front.
    let seed: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let input = client.create_from_slice(f32::as_bytes(&seed));
    let scratch = [
        client.empty(N * core::mem::size_of::<f32>()),
        client.empty(N * core::mem::size_of::<f32>()),
        client.empty(N * core::mem::size_of::<f32>()),
    ];
    let output = client.empty(N * core::mem::size_of::<f32>());

    // ---- Phase 1: eager ----
    // One untimed pass first: the very first launch compiles the kernels, which costs
    // hundreds of milliseconds and has nothing to do with per-step dispatch cost.
    decode_step::<R>(&client, &input, &scratch, &output);
    read_f32::<R>(&client, &output)?;

    let started = Instant::now();
    for _ in 0..STEPS {
        decode_step::<R>(&client, &input, &scratch, &output);
    }
    let eager = read_f32::<R>(&client, &output)?;
    let eager_elapsed = started.elapsed();
    println!(
        "eager:    {STEPS} steps ({} launches) in {eager_elapsed:?}, output[0]={:.1}",
        STEPS * launches,
        eager[0]
    );

    // ---- Phase 2: captured ----
    client
        .graph_prepare()
        .map_err(|e| format!("graph_prepare failed: {e:?}"))?;

    // Rule 2: warm up with the exact closure that will be recorded.
    for _ in 0..WARMUP {
        decode_step::<R>(&client, &input, &scratch, &output);
    }
    future::block_on(client.sync()).map_err(|e| format!("warmup sync failed: {e:?}"))?;

    client
        .start_capture()
        .map_err(|e| format!("start_capture failed: {e:?}"))?;
    // Recorded, not executed.
    decode_step::<R>(&client, &input, &scratch, &output);
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
        "replayed: {STEPS} steps ({} launches) in {replay_elapsed:?}, output[0]={:.1}",
        STEPS * launches,
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
    let expected = bumped[0] + LAYERS as f32;
    if (after[0] - expected).abs() > 1e-3 {
        return Err(format!(
            "input refresh not observed: expected {expected:.1}, got {:.1}",
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
/// buffers are allocated. They are created once here, at fixed capacity, before
/// `graph_prepare` — so the recorded window only ever reuses them. Allocating per layer
/// per step (inside the window) would record memory nodes instead, and the driver would
/// refuse to relaunch the resulting graph past the first replay. Nothing else about a
/// state-carrying workload makes it hostile to capture.
pub fn stateful<R: Runtime>(device: &R::Device) -> Res {
    let client = R::client(device);
    let launches = 2 * LAYERS;
    println!("== stateful graph capture on {:?} ==", R::name(&client));
    println!("   {LAYERS} layers, {launches} launches per step, {STEPS} steps");

    let ones = vec![1.0f32; N];
    let zeros = vec![0.0f32; N];

    // Allocated once, at fixed capacity, BEFORE graph_prepare — one state buffer per
    // layer, exactly as a real per-layer cache would be. This is the rule.
    let input = client.create_from_slice(f32::as_bytes(&ones));
    let state: Vec<Handle> = (0..LAYERS)
        .map(|_| client.create_from_slice(f32::as_bytes(&zeros)))
        .collect();
    let tmp = client.empty(N * core::mem::size_of::<f32>());
    let act = client.empty(N * core::mem::size_of::<f32>());
    let output = client.empty(N * core::mem::size_of::<f32>());

    let reset_state = |client: &ComputeClient<R>| {
        for buffer in &state {
            write_f32::<R>(client, buffer, &zeros);
        }
    };

    // ---- Phase 1: eager ----
    // Untimed pass to compile the kernels; it advances the state, so reset afterwards.
    recurrent_step::<R>(&client, &input, &state, &tmp, &act, &output);
    read_f32::<R>(&client, &output)?;
    reset_state(&client);

    let started = Instant::now();
    for _ in 0..STEPS {
        recurrent_step::<R>(&client, &input, &state, &tmp, &act, &output);
    }
    let eager = read_f32::<R>(&client, &output)?;
    let eager_elapsed = started.elapsed();
    println!(
        "eager:    {STEPS} steps ({} launches) in {eager_elapsed:?}, output[0]={:.1}",
        STEPS * launches,
        eager[0]
    );

    // ---- Phase 2: captured ----
    reset_state(&client);

    client
        .graph_prepare()
        .map_err(|e| format!("graph_prepare failed: {e:?}"))?;

    // Warmup mutates the state, so reset again afterwards; what matters is that warmup
    // runs the identical launches, priming the pool for the window.
    for _ in 0..WARMUP {
        recurrent_step::<R>(&client, &input, &state, &tmp, &act, &output);
    }
    future::block_on(client.sync()).map_err(|e| format!("warmup sync failed: {e:?}"))?;
    reset_state(&client);

    client
        .start_capture()
        .map_err(|e| format!("start_capture failed: {e:?}"))?;
    // Recorded, not executed — so it does NOT advance the state.
    recurrent_step::<R>(&client, &input, &state, &tmp, &act, &output);
    let graph = client
        .stop_capture()
        .map_err(|e| format!("stop_capture failed: {e:?}"))?;

    let started = Instant::now();
    for _ in 0..STEPS {
        // SAFETY: input, state, tmp, act and output are all still alive above, and every
        // replay and read stays on this one client's stream.
        unsafe { graph.replay() };
    }
    let replayed = read_f32::<R>(&client, &output)?;
    let replay_elapsed = started.elapsed();
    println!(
        "replayed: {STEPS} steps ({} launches) in {replay_elapsed:?}, output[0]={:.1}",
        STEPS * launches,
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

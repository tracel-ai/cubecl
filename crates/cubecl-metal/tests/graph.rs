//! Validates Metal graph capture/replay on the actual device.
//!
//! The recorded dispatches are encoded into an indirect command buffer when
//! the window closes, and a replay executes it with one call; the lifecycle
//! contract is the CUDA/HIP one: `graph_prepare`, run the workload once,
//! `start_capture`, run it again (recorded, not executed), `stop_capture`,
//! then `replay`.
#![cfg(target_vendor = "apple")]

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::capture_status;
use cubecl_core::server::Handle;
use cubecl_environment::stream::StreamId;
use cubecl_metal::MetalRuntime;
use cubecl_server::runtime::Runtime;
use std::sync::Mutex;

/// Test threads share the cached client, and the stream pool can map two test
/// threads' stream ids onto the same backend stream — where two concurrent
/// captures would reject each other. One capture at a time, as in real use.
static CAPTURE_LOCK: Mutex<()> = Mutex::new(());

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
fn add_one_tensor(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// `output = input + 1` over four elements, launched on `client`.
fn launch_add_one(client: &Client, input: &Handle, output: &Handle) {
    let n = 4usize;
    add_one::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(client, n),
        unsafe { BufferArg::from_raw_parts(input.clone(), n) },
        unsafe { BufferArg::from_raw_parts(output.clone(), n) },
    );
}

/// A fresh `[1, 2, 3, 4]` input and an output of the same size.
fn input_and_output(client: &Client) -> (Handle, Handle) {
    (
        client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0])),
        client.empty(4 * core::mem::size_of::<f32>()),
    )
}

/// Capture a single kernel launch into a graph, replay it, and check the
/// output — the end-to-end proof that graph capture works on this GPU.
#[test]
fn metal_graph_capture_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);

    // Prepare starts the warmup run; it is mandatory before a capture.
    client.graph_prepare().expect("graph_prepare");
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    // Record one launch into a graph instead of executing it.
    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    let graph = client.stop_capture().expect("stop_capture");

    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Replaying again re-runs it deterministically.
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A capture window that has to grow the memory pool is rejected, not handed
/// back: every replay has to use the buffers the warmup left.
#[test]
fn metal_graph_capture_growing_the_pool_is_rejected() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);

    client.graph_prepare().expect("graph_prepare");
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    // This deliberately odd size has never been allocated on this client, so
    // no free slice fits it and the storage must reach the device allocator.
    let grown = client.empty(3_145_733);
    assert!(
        client.stop_capture().is_err(),
        "stop_capture must reject a capture that tried to grow the memory pool"
    );

    drop(grown);
}

/// A captured graph reads its input buffer at replay time, so writing new
/// bytes into that same buffer and replaying produces output for the new
/// input — how a decode loop feeds the next token into a captured step.
#[test]
fn metal_graph_input_rewrite() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);

    client.graph_prepare().expect("graph_prepare");
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    let graph = client.stop_capture().expect("stop_capture");

    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]).to_vec()),
    );
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[11.0, 21.0, 31.0, 41.0]);
}

/// A graph's intermediate buffer is allocated inside the window; dropping its
/// handle afterwards must not let the pool hand its slice to a later
/// allocation, which every replay would then overwrite.
///
/// Computes `(input + 1) * 2` as two kernels through `tmp`, drops `tmp`,
/// allocates sentinels, then replays: the output is right and no sentinel was
/// written.
#[test]
fn metal_graph_intermediate_recycling() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let (input, output) = input_and_output(&client);

    let run = |client: &Client, tmp: &Handle| {
        launch_add_one(client, &input, tmp);
        mul_two::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    {
        let tmp = client.empty(bytes);
        run(&client, &tmp);
        let _ = client.read_one(output.clone()).unwrap();
    }

    client.start_capture().expect("start_capture");
    let tmp = client.empty(bytes);
    run(&client, &tmp);
    let graph = client.stop_capture().expect("stop_capture");

    drop(tmp);
    let sentinels: Vec<Handle> = (0..8)
        .map(|_| client.create_from_slice(f32::as_bytes(&[999.0; 4])))
        .collect();

    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(
        f32::from_bytes(&out),
        &[4.0, 6.0, 8.0, 10.0],
        "graph output corrupted"
    );

    let clobbered = sentinels.iter().any(|handle| {
        let bytes = client.read_one(handle.clone()).unwrap();
        f32::from_bytes(&bytes) == [2.0, 3.0, 4.0, 5.0]
    });
    assert!(
        !clobbered,
        "replay wrote into a live buffer that reused the graph's intermediate slice"
    );
}

/// Decode-shaped stress: a window of many dependent launches of a `Tensor`
/// kernel that reads `shape(0)`, so every launch carries metadata, past the
/// stream's batch limits. The recorded pass does not execute, two replays
/// re-run it exactly — in order, each launch reading what the one before it
/// wrote — and an in-place input rewrite feeds the next replay.
#[test]
fn metal_graph_many_launches_dynamic_metadata() {
    const N: usize = 64;
    const PASS_LAUNCHES: usize = 150;

    // One pass: ping-pong `dst = src + 1` between `a` and `b`.
    fn run_pass(client: &Client, a: &Handle, b: &Handle) {
        for i in 0..PASS_LAUNCHES {
            let (src, dst) = if i % 2 == 0 { (a, b) } else { (b, a) };
            add_one_tensor::launch(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(N as u32),
                // SAFETY: `src`/`dst` are contiguous rank-1 buffers of exactly
                // `N` f32 elements, matching the declared shape and strides.
                unsafe { TensorArg::from_raw_parts(src.clone(), [1].into(), [N].into()) },
                unsafe { TensorArg::from_raw_parts(dst.clone(), [1].into(), [N].into()) },
            );
        }
    }

    // What `a` and `b` hold after `passes` executed passes from equal starts.
    fn simulate(start: f32, passes: usize) -> (Vec<f32>, Vec<f32>) {
        let launches = (passes * PASS_LAUNCHES) as f32;
        // `b` was last written by the final even launch, `a` by the final odd.
        (vec![start + launches; N], vec![start + launches - 1.0; N])
    }

    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let read = |handle: &Handle| f32::from_bytes(&client.read_one(handle.clone()).unwrap()).to_vec();

    let a = client.create_from_slice(f32::as_bytes(&[0.0f32; N]));
    let b = client.create_from_slice(f32::as_bytes(&[0.0f32; N]));

    client.graph_prepare().expect("graph_prepare");
    run_pass(&client, &a, &b);
    assert_eq!((read(&a), read(&b)), simulate(0.0, 1));

    client.start_capture().expect("start_capture");
    run_pass(&client, &a, &b);
    let graph = client.stop_capture().expect("stop_capture");

    // Warmup + 2 replays = 3 executed passes (the recorded pass ran 0 times).
    unsafe { graph.replay() }.expect("replay enqueues");
    unsafe { graph.replay() }.expect("replay enqueues");
    assert_eq!((read(&a), read(&b)), simulate(0.0, 3));

    let fresh = f32::as_bytes(&[100.0f32; N]).to_vec();
    client.write(&a, Bytes::from_bytes_vec(fresh.clone()));
    client.write(&b, Bytes::from_bytes_vec(fresh));
    unsafe { graph.replay() }.expect("replay enqueues");
    assert_eq!((read(&a), read(&b)), simulate(100.0, 1));
}

/// Out-of-order lifecycle calls are rejected and leave the stream usable.
#[test]
fn metal_graph_lifecycle_state_errors() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());

    assert!(
        client.start_capture().is_err(),
        "start_capture without graph_prepare must be rejected"
    );
    assert!(
        client.stop_capture().is_err(),
        "stop_capture without a recording capture must be rejected"
    );

    client.graph_prepare().expect("graph_prepare");
    assert!(
        client.graph_prepare().is_err(),
        "a second graph_prepare on a prepared stream must be rejected"
    );

    let (input, output) = input_and_output(&client);
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    let graph = client.stop_capture().expect("stop_capture");
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A read inside a recording window is rejected directly — the recorded
/// launches have not run, so there is nothing true to hand back — and the
/// capture still completes afterwards.
#[test]
fn metal_graph_read_rejected_while_recording() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);

    client.graph_prepare().expect("graph_prepare");
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    assert!(
        client.read_one(output.clone()).is_err(),
        "a read inside a capture window must be rejected"
    );

    let graph = client.stop_capture().expect("stop_capture");
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A write inside a recording window would run once instead of on every
/// replay, so it is rejected and `stop_capture` fails, rather than handing
/// back a graph that silently skips it.
#[test]
fn metal_graph_write_rejected_while_recording() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);

    client.graph_prepare().expect("graph_prepare");
    launch_add_one(&client, &input, &output);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch_add_one(&client, &input, &output);
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[9.0, 9.0, 9.0, 9.0]).to_vec()),
    );
    assert!(
        client.stop_capture().is_err(),
        "a capture window containing a write must be rejected"
    );

    // The rejected write left its destination carrying the rejection, so the
    // stream is usable again once the input is written for real.
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]).to_vec()),
    );
    launch_add_one(&client, &input, &output);
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A dynamic cube count is read on the host, and inside a recording window
/// that read has nothing true to read: the count a replay needs is not known
/// until it runs. The window is doomed rather than sealed with the count the
/// buffer held while recording.
#[test]
fn metal_graph_dynamic_count_rejected_while_recording() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let (input, output) = input_and_output(&client);
    let count = client.create_from_slice(u32::as_bytes(&[1, 1, 1]));
    let launch = |client: &Client| {
        add_one::launch(
            client,
            CubeCount::Dynamic(count.clone().binding()),
            CubeDim::new(client, 4),
            unsafe { BufferArg::from_raw_parts(input.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output.clone(), 4) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    client.start_capture().expect("start_capture");
    launch(&client);
    assert!(
        client.stop_capture().is_err(),
        "a capture window containing a dynamic count must be rejected"
    );
}

/// A capture that never seals leaves its recorded launches unrun, so a read of
/// what they were given — from a stream that was never told — fails rather
/// than handing back the bytes from before, until the owner writes them again.
#[test]
fn metal_graph_a_capture_that_never_sealed_is_not_read_as_run() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = MetalRuntime::client(&Default::default());
    let capturing = StreamId { value: 4_000_001 };
    let reader = StreamId { value: 4_000_002 };

    let (input, output, target) = capturing.executes(|| {
        let (input, output) = input_and_output(&client);
        (
            input,
            output,
            client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0])),
        )
    });

    capturing.executes(|| {
        client.graph_prepare().expect("graph_prepare");
        launch_add_one(&client, &input, &output);
        let _ = client.read_one(output.clone()).unwrap();
        client.start_capture().expect("start_capture");
        launch_add_one(&client, &input, &output);
        // Rejected inside the window, so the capture cannot seal.
        client.write(
            &target,
            Bytes::from_bytes_vec(f32::as_bytes(&[9.0, 9.0, 9.0, 9.0]).to_vec()),
        );
    });
    assert!(
        capturing.executes(|| client.stop_capture()).is_err(),
        "a capture window containing a write must be rejected"
    );

    assert!(
        reader.executes(|| client.read_one(output.clone())).is_err(),
        "the recorded launch never ran, so reading its output must fail"
    );

    let out = capturing.executes(|| {
        launch_add_one(&client, &input, &output);
        client.read_one(output.clone()).unwrap()
    });
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
    let out = reader.executes(|| client.read_one(output.clone()).unwrap());
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// See [`capture_status::a_capture_is_seen_by_every_client_of_its_stream`].
#[test]
fn metal_graph_capture_is_seen_by_every_client_of_its_stream() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    capture_status::a_capture_is_seen_by_every_client_of_its_stream::<MetalRuntime>();
}

/// See [`capture_status::a_prepared_capture_ends_at_stop_capture`].
#[test]
fn metal_graph_prepared_capture_ends_at_stop_capture() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    capture_status::a_prepared_capture_ends_at_stop_capture::<MetalRuntime>();
}

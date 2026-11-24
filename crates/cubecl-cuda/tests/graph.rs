//! Validates CUDA graph capture/replay on the actual device.

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::runtime_tests::{capture_status, graph_sharing};
use cubecl_core::server::Handle;
use cubecl_cuda::CudaRuntime;
use cubecl_server::runtime::Runtime;
use std::sync::Mutex;

/// Graph capture holds the one cached client's stream for its whole window,
/// and `CU_STREAM_CAPTURE_MODE_GLOBAL` makes any
/// concurrent unsafe call (alloc, sync) in the process abort a recording
/// capture — so two captures must not overlap: exactly one capture at a time
/// per device, as in real use. Serialize the tests instead of relying on
/// `--test-threads 1`.
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

/// Capture a single kernel launch into a graph, replay it, and check the
/// output — the end-to-end proof that hardware graph capture works on this GPU.
#[test]
fn cuda_graph_capture_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &Client| {
        add_one::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Plan the launch (compiled, discarded), then record it into a graph
    // instead of executing it.
    let (graph, ()) = client.capture(|| launch(&client)).expect("capture");

    // Replay executes the recorded launch; the output is input + 1.
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Replaying again re-runs it deterministically.
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A capture window that has to grow the memory pool must be REJECTED, not handed back.
///
/// The server rejects storage growth while recording, so every replay uses buffers
/// the planning run left. Force that guard by requesting an allocation larger than
/// the pool has served before, then verify that capture is rejected.
#[test]
fn cuda_graph_capture_growing_the_pool_is_rejected() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &Client| {
        add_one::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    client.graph_end_run().expect("graph_end_run");
    client.start_capture().expect("start_capture");
    launch(&client);
    // Force the pool to grow mid-capture: this deliberately odd size has never been allocated on
    // this client, so no free slice fits it and the storage must reach the device allocator.
    let grown = client.empty(3_145_733);
    let rejected = client.stop_capture();

    assert!(
        rejected.is_err(),
        "stop_capture must reject a capture that tried to grow the memory pool"
    );

    drop(grown);
}

/// The input-rewrite path: a captured graph reads its input buffer at replay
/// time, so writing new bytes into that same buffer (same device pointer) and
/// replaying must produce output for the new input. This is how a decode loop
/// feeds the next token into a captured step without re-capturing.
#[test]
fn cuda_graph_input_rewrite() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &Client| {
        add_one::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    let (graph, ()) = client.capture(|| launch(&client)).expect("capture");

    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Write new inputs into the captured buffer (same pointer), replay: the
    // output must reflect the new input, not the captured-time values.
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]).to_vec()),
    );
    unsafe { graph.replay() }.expect("replay enqueues");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[11.0, 21.0, 31.0, 41.0]);
}

/// The lifecycle risk: a captured graph holds raw device pointers the memory
/// pool cannot see. Its **intermediate** buffer (`tmp`) is allocated during
/// capture; when its handle drops, does the pool reclaim that memory and hand
/// it to a later allocation, corrupting a replay?
///
/// Computes `(input + 1) * 2` as two kernels through `tmp`, drops `tmp`,
/// reallocates sentinel buffers over its freed slice, then replays.
///
/// The graph's own output stays correct (its first kernel rewrites `tmp`
/// before the second reads it — write-before-read), and because `end_capture`
/// guards every page the recording touched, a later allocation can no longer
/// reuse `tmp`'s slice, so replay does **not** clobber the sentinels. This is
/// the acceptance test for that guard.
#[test]
fn cuda_graph_intermediate_recycling() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(bytes);

    let run = |client: &Client, tmp: &Handle| {
        add_one::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
        );
        mul_two::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Capture the two-kernel computation. `tmp` is its temporary: allocated
    // and dropped by every run, so the recorded run reuses the slice the
    // planning runs left. The graph still references its pointer, but the pool
    // now thinks the slice is free — so reallocate over it with sentinels.
    let (graph, ()) = client
        .capture(|| {
            let tmp = client.empty(bytes);
            run(&client, &tmp);
        })
        .expect("capture");
    let sentinels: Vec<Handle> = (0..8)
        .map(|_| client.create_from_slice(f32::as_bytes(&[999.0; 4])))
        .collect();

    // Replay. The graph's own OUTPUT is correct regardless: its first kernel
    // rewrites `tmp` before the second reads it (write-before-read), so
    // external reuse cannot corrupt the graph's result.
    unsafe { graph.replay() }.expect("replay enqueues");
    let out_bytes = client.read_one(output).unwrap();
    let out = f32::from_bytes(&out_bytes);
    println!("graph output: {out:?} (want [4, 6, 8, 10])");
    assert_eq!(out, &[4.0, 6.0, 8.0, 10.0], "graph output corrupted");

    // The real hazard is the other direction: if a sentinel was placed on the
    // graph's freed `tmp` slice, the replay's first kernel WROTE `input + 1`
    // into it — clobbering the sentinel. If any sentinel now reads `[2,3,4,5]`
    // instead of `[999,999,999,999]`, replay corrupted live external memory.
    let clobbered = sentinels.iter().any(|h| {
        let bytes = client.read_one(h.clone()).unwrap();
        f32::from_bytes(&bytes) == [2.0, 3.0, 4.0, 5.0]
    });
    println!("a sentinel buffer was clobbered by replay: {clobbered}");
    assert!(
        !clobbered,
        "replay wrote into a live external buffer that reused the graph's \
         intermediate slice — buffer retention failed to pin it"
    );
}

#[cube(launch)]
fn add_one_tensor(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// Decode-shaped stress: capture a window of many launches (well past the
/// drop-queue flush threshold of 64 pushes, so the deferred-flush/staging-priming
/// path is exercised) of a `Tensor` kernel that reads `shape(0)`, forcing
/// every launch through the dynamic-metadata staging + info-cache path. Then
/// verify the recorded pass did not execute, two replays re-run it exactly,
/// and an in-place input rewrite feeds the next replay.
#[test]
fn cuda_graph_many_launches_dynamic_metadata() {
    const N: usize = 64; // elements per tensor; one 64-thread block covers them
    const PASS_LAUNCHES: usize = 150; // > 2x the drop-queue flush threshold

    // One pass: ping-pong `dst = src + 1` between `a` and `b`. The identical
    // sequence is planned (discarded) and then recorded.
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
        let (mut a, mut b) = (vec![start; N], vec![start; N]);
        for _ in 0..passes {
            for i in 0..PASS_LAUNCHES {
                if i % 2 == 0 {
                    for j in 0..N {
                        b[j] = a[j] + 1.0;
                    }
                } else {
                    for j in 0..N {
                        a[j] = b[j] + 1.0;
                    }
                }
            }
        }
        (a, b)
    }

    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());

    let a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));

    // Plan and record the pass; neither the planning runs nor the recorded
    // run execute it.
    let (graph, ()) = client
        .capture(|| run_pass(&client, &a, &b))
        .expect("capture");

    // 2 replays = 2 executed passes.
    unsafe { graph.replay() }.expect("replay enqueues");
    unsafe { graph.replay() }.expect("replay enqueues");
    let (exp_a, exp_b) = simulate(0.0, 2);
    assert_eq!(
        f32::from_bytes(&client.read_one(a.clone()).unwrap()),
        &exp_a[..]
    );
    assert_eq!(
        f32::from_bytes(&client.read_one(b.clone()).unwrap()),
        &exp_b[..]
    );

    // In-place input rewrite (the decode-loop pattern), then one more replay:
    // the graph must compute from the new values.
    let fresh = f32::as_bytes(&[100.0f32; N]).to_vec();
    client.write(&a, Bytes::from_bytes_vec(fresh.clone()));
    client.write(&b, Bytes::from_bytes_vec(fresh));
    unsafe { graph.replay() }.expect("replay enqueues");
    let (exp_a, exp_b) = simulate(100.0, 1);
    assert_eq!(
        f32::from_bytes(&client.read_one(a.clone()).unwrap()),
        &exp_a[..]
    );
    assert_eq!(
        f32::from_bytes(&client.read_one(b.clone()).unwrap()),
        &exp_b[..]
    );
}

/// See [`capture_status::a_capture_is_seen_by_every_client_of_its_stream`].
#[test]
fn cuda_graph_capture_is_seen_by_every_client_of_its_stream() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    capture_status::a_capture_is_seen_by_every_client_of_its_stream::<CudaRuntime>();
}

/// See [`capture_status::a_prepared_capture_ends_at_stop_capture`].
#[test]
fn cuda_graph_prepared_capture_ends_at_stop_capture() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    capture_status::a_prepared_capture_ends_at_stop_capture::<CudaRuntime>();
}

/// See [`graph_sharing::a_smaller_graph_fits_in_the_pages_a_larger_one_pinned`].
#[test]
fn cuda_graph_smaller_graph_fits_in_the_pages_a_larger_one_pinned() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::a_smaller_graph_fits_in_the_pages_a_larger_one_pinned::<CudaRuntime>();
}

/// See [`graph_sharing::graphs_sharing_pinned_pages_replay_in_any_order`].
#[test]
fn cuda_graph_graphs_sharing_pinned_pages_replay_in_any_order() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::graphs_sharing_pinned_pages_replay_in_any_order::<CudaRuntime>();
}

/// See [`graph_sharing::a_held_output_survives_another_graphs_replay`].
#[test]
fn cuda_graph_held_output_survives_another_graphs_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::a_held_output_survives_another_graphs_replay::<CudaRuntime>();
}

/// See [`graph_sharing::an_output_survives_an_earlier_graphs_replay`].
#[test]
fn cuda_graph_output_survives_an_earlier_graphs_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::an_output_survives_an_earlier_graphs_replay::<CudaRuntime>();
}

/// See [`graph_sharing::a_recording_that_keeps_a_planned_temporary_does_not_seal`].
#[test]
fn cuda_graph_recording_that_keeps_a_planned_temporary_does_not_seal() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::a_recording_that_keeps_a_planned_temporary_does_not_seal::<CudaRuntime>();
}

/// See [`graph_sharing::outputs_pack_apart_from_the_shared_room`].
#[test]
fn cuda_graph_outputs_pack_apart_from_the_shared_room() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::outputs_pack_apart_from_the_shared_room::<CudaRuntime>();
}

/// See [`capture_status::a_work_that_panics_leaves_no_capture`].
#[test]
fn cuda_graph_work_that_panics_leaves_no_capture() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    capture_status::a_work_that_panics_leaves_no_capture::<CudaRuntime>();
}

/// See [`graph_sharing::an_input_written_between_runs_survives_another_graphs_replay`].
#[test]
fn cuda_graph_input_written_between_runs_survives_another_graphs_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::an_input_written_between_runs_survives_another_graphs_replay::<CudaRuntime>();
}

/// See [`graph_sharing::inputs_kept_per_graph_take_no_page_of_their_own`].
#[test]
fn cuda_graph_inputs_kept_per_graph_take_no_page_of_their_own() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::inputs_kept_per_graph_take_no_page_of_their_own::<CudaRuntime>();
}

/// See [`graph_sharing::small_buffers_kept_per_graph_take_no_page_of_their_own`].
#[test]
fn cuda_graph_small_buffers_kept_per_graph_take_no_page_of_their_own() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::small_buffers_kept_per_graph_take_no_page_of_their_own::<CudaRuntime>();
}

/// See [`graph_sharing::a_recording_allocating_a_temporary_its_plan_did_not_seals`].
#[test]
fn cuda_graph_recording_allocating_a_temporary_its_plan_did_not_seals() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::a_recording_allocating_a_temporary_its_plan_did_not_seals::<CudaRuntime>();
}

/// See [`graph_sharing::a_recording_skipping_a_temporary_its_plan_made_seals`].
#[test]
fn cuda_graph_recording_skipping_a_temporary_its_plan_made_seals() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    graph_sharing::a_recording_skipping_a_temporary_its_plan_made_seals::<CudaRuntime>();
}

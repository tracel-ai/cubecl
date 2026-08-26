//! Validates wgpu graph capture/replay on the actual device.
//!
//! The wgpu graph is a software graph (recorded, fully-resolved dispatches
//! re-encoded on replay), but the lifecycle contract is the CUDA/HIP one:
//! warm autotune, `graph_prepare`, run the workload once, `start_capture`,
//! run it again (recorded, not executed), `stop_capture`, then `replay`.

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_wgpu::WgpuRuntime;
use std::sync::Mutex;

/// Test threads share the cached client, and the scheduler's stream pool can
/// map two test threads' stream ids onto the same backend stream — where two
/// concurrent captures would reject each other. One capture at a time, as in
/// real use.
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
/// output — the end-to-end proof that graph capture works on this GPU.
#[test]
fn wgpu_graph_capture_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Prepare arms the persistent pools; it is mandatory before a capture.
    client.graph_prepare().expect("graph_prepare");

    // Warm up: compile the kernel and allocate every buffer, so the capture
    // run stays on the warm path.
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    // Record one launch into a graph instead of executing it.
    client.start_capture().expect("start_capture");
    launch(&client);
    let graph = client.stop_capture().expect("stop_capture");

    // Replay executes the recorded launch; the output is input + 1.
    unsafe { graph.replay() };
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Replaying again re-runs it deterministically.
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A capture window that allocates fresh memory is fine on wgpu — the
/// opposite of CUDA/HIP, where a mid-capture allocation records a memory node
/// that makes the graph un-relaunchable. A software graph has no such
/// constraint: the fresh slice is simply pinned to the graph like everything
/// else the window touched.
#[test]
fn wgpu_graph_mid_capture_allocation_is_allowed() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);
    // A size no warmup allocated and no pool bucket rounds to, so serving it
    // forces the pool to grow *inside* the window — the thing under test.
    // Deliberately not a round number: a power of two would land in an
    // existing bucket and the window would allocate nothing.
    const UNPOOLED_BYTES: usize = 3_145_733;
    let grown = client.empty(UNPOOLED_BYTES);
    let graph = client.stop_capture().expect(
        "a mid-capture allocation is legal on wgpu: the fresh slice is pinned to the graph",
    );

    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    drop(grown);
}

/// The input-rewrite path: a captured graph reads its input buffer at replay
/// time, so writing new bytes into that same buffer and replaying must produce
/// output for the new input. This is how a decode loop feeds the next token
/// into a captured step without re-capturing.
#[test]
fn wgpu_graph_input_rewrite() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");

    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);
    let graph = client.stop_capture().expect("stop_capture");

    unsafe { graph.replay() };
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Write new inputs into the captured buffer (same slice), replay: the
    // output must reflect the new input, not the captured-time values.
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]).to_vec()),
    );
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[11.0, 21.0, 31.0, 41.0]);
}

/// The lifecycle risk: a captured graph references pool slices the memory
/// pool cannot see. Its **intermediate** buffer (`tmp`) is allocated during
/// capture; when its handle drops, does the pool reclaim that memory and hand
/// it to a later allocation, corrupting a replay?
///
/// Computes `(input + 1) * 2` as two kernels through `tmp`, drops `tmp`,
/// reallocates sentinel buffers over its freed slice, then replays.
///
/// The graph's own output stays correct (its first kernel rewrites `tmp`
/// before the second reads it — write-before-read), and with buffer retention
/// (`graph_prepare` routes capture-phase allocations into the persistent
/// pools, warmup populates them, `end_capture` pins those slices) a later
/// allocation can no longer reuse `tmp`'s slice, so replay does **not**
/// clobber the sentinels. This is the acceptance test for that retention.
#[test]
fn wgpu_graph_intermediate_recycling() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(bytes);

    let run = |client: &ComputeClient<WgpuRuntime>, tmp: &Handle| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
        );
        mul_two::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Prepare: capture-phase allocations now go to the persistent pools and
    // are tracked for retention.
    client.graph_prepare().expect("graph_prepare");

    // Warm up so `tmp` is compiled + allocated in the persistent pool (then
    // freed, so the capture run reuses it without a fresh allocation).
    {
        let tmp = client.empty(bytes);
        run(&client, &tmp);
        let _ = client.read_one(output.clone()).unwrap();
    }

    // Capture the two-kernel computation; `tmp` reuses the warm persistent slice.
    client.start_capture().expect("start_capture");
    let tmp = client.empty(bytes);
    run(&client, &tmp);
    let graph = client.stop_capture().expect("stop_capture");

    // Drop `tmp` — the graph still references its slice, but the pool now
    // thinks the slice is free — then reallocate over it with sentinel buffers.
    drop(tmp);
    let sentinels: Vec<Handle> = (0..8)
        .map(|_| client.create_from_slice(f32::as_bytes(&[999.0; 4])))
        .collect();

    // Replay. The graph's own OUTPUT is correct regardless: its first kernel
    // rewrites `tmp` before the second reads it (write-before-read), so
    // external reuse cannot corrupt the graph's result.
    unsafe { graph.replay() };
    let out_bytes = client.read_one(output).unwrap();
    let out = f32::from_bytes(&out_bytes);
    assert_eq!(
        out,
        &[4.0, 6.0, 8.0, 10.0],
        "the graph's own output is corrupted, before the retention question"
    );

    // The real hazard is the other direction: if a sentinel was placed on the
    // graph's freed `tmp` slice, the replay's first kernel WROTE `input + 1`
    // into it — clobbering the sentinel. If any sentinel now reads `[2,3,4,5]`
    // instead of `[999,999,999,999]`, replay corrupted live external memory.
    let clobbered = sentinels.iter().any(|h| {
        let bytes = client.read_one(h.clone()).unwrap();
        f32::from_bytes(&bytes) == [2.0, 3.0, 4.0, 5.0]
    });
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
/// scheduler's max-tasks threshold, so mid-window drains of the task queue
/// are exercised) of a `Tensor` kernel that reads `shape(0)`, forcing every
/// launch through the metadata info-cache path. Then verify the recorded pass
/// did not execute, two replays re-run it exactly, and an in-place input
/// rewrite feeds the next replay.
#[test]
fn wgpu_graph_many_launches_dynamic_metadata() {
    const N: usize = 64; // elements per tensor; one 64-thread cube covers them
    const PASS_LAUNCHES: usize = 150;

    // One pass: ping-pong `dst = src + 1` between `a` and `b`. The identical
    // sequence is run once as warmup and once recorded.
    fn run_pass(client: &ComputeClient<WgpuRuntime>, a: &Handle, b: &Handle) {
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
    let client = WgpuRuntime::client(&Default::default());

    let a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));
    let b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; N]));

    client.graph_prepare().expect("graph_prepare");
    run_pass(&client, &a, &b);

    // The warmup executed normally (reads are still legal while prepared).
    let (exp_a, exp_b) = simulate(0.0, 1);
    assert_eq!(
        f32::from_bytes(&client.read_one(a.clone()).unwrap()),
        &exp_a[..]
    );
    assert_eq!(
        f32::from_bytes(&client.read_one(b.clone()).unwrap()),
        &exp_b[..]
    );

    // Record the identical sequence; recorded launches must not execute.
    client.start_capture().expect("start_capture");
    run_pass(&client, &a, &b);
    let graph = client.stop_capture().expect("stop_capture");

    // Warmup + 2 replays = 3 executed passes (the recorded pass ran 0 times).
    unsafe { graph.replay() };
    unsafe { graph.replay() };
    let (exp_a, exp_b) = simulate(0.0, 3);
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
    unsafe { graph.replay() };
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

/// Out-of-order lifecycle calls are rejected and leave the stream usable.
#[test]
fn wgpu_graph_lifecycle_state_errors() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());

    // No capture prepared or recording.
    assert!(
        client.start_capture().is_err(),
        "start_capture without graph_prepare must be rejected"
    );
    assert!(
        client.stop_capture().is_err(),
        "stop_capture without a recording capture must be rejected"
    );

    // Double prepare.
    client.graph_prepare().expect("graph_prepare");
    assert!(
        client.graph_prepare().is_err(),
        "a second graph_prepare on a prepared stream must be rejected"
    );

    // The stream still works: a normal capture goes through end to end.
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());
    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);
    let graph = client.stop_capture().expect("stop_capture");
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// Host syncs cannot happen inside a recording window: a read is rejected
/// directly (returning an error, not wedging the stream), and the capture can
/// still be completed afterwards.
#[test]
fn wgpu_graph_read_rejected_while_recording() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());
    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);
    assert!(
        client.read_one(output.clone()).is_err(),
        "a read inside a capture window must be rejected"
    );

    // The rejection did not poison the capture: it completes and replays.
    let graph = client.stop_capture().expect("stop_capture");
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// Writing data inside a recording window is unsupported on wgpu (v1): the
/// write is rejected lazily and `stop_capture` fails, rather than handing back
/// a graph that silently skips the write.
#[test]
fn wgpu_graph_write_rejected_while_recording() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());
    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[9.0, 9.0, 9.0, 9.0]).to_vec()),
    );
    assert!(
        client.stop_capture().is_err(),
        "a capture window containing a write must be rejected: the recording \
         is missing that operation"
    );

    // The failed capture left the stream usable.
    launch(&client);
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// Destroying a graph with a replay still in flight does not corrupt that
/// replay, even though the destroy hands the graph's memory straight back to
/// the pool for the next allocation to take.
///
/// A replay is *encoded*, not submitted, and `queue.write_buffer` runs at the
/// next submit ahead of everything already in the encoder — so a write onto
/// reclaimed memory can reach the GPU before the replay that reads it. The
/// contract this defends is the one a decode loop depends on: dropping the
/// last `Graph` handle is safe at any point, and the work already enqueued
/// still computes what was captured.
///
/// This is a contract test, not a regression test for one line: the `Write`
/// path flushes before writing on its own account, so it holds even with
/// `graph_destroy`'s own submit removed. That submit covers the paths that do
/// not flush first (uniform writes), whose reuse of a specific released slice
/// is not something a test can force.
#[test]
fn wgpu_graph_destroy_leaves_an_enqueued_replay_intact() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(bytes);

    let run = |client: &ComputeClient<WgpuRuntime>, tmp: &Handle| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
        );
        mul_two::launch::<WgpuRuntime>(
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
    // Only the graph pins this slice now.
    drop(tmp);

    // Encode the replay, then destroy without reading: the submit that makes
    // the replay real has to come from `graph_destroy`.
    unsafe { graph.replay() };
    drop(graph);

    // Reallocate over the just-released memory. Each of these writes would
    // overtake an unsubmitted replay.
    let _sentinels: Vec<Handle> = (0..8)
        .map(|_| client.create_from_slice(f32::as_bytes(&[999.0; 4])))
        .collect();

    let out_bytes = client.read_one(output).unwrap();
    assert_eq!(
        f32::from_bytes(&out_bytes),
        &[4.0, 6.0, 8.0, 10.0],
        "the enqueued replay read sentinel bytes: destroying the graph let a \
         later write reach its memory first"
    );
}

/// A capture window is closed by the stream that opened it, and by no other.
///
/// The errors raised inside the window — a rejected write, a failed binding —
/// are queued for the stream that opened it, and `end_capture` drains them to
/// decide whether the recording is complete. A neighbour closing the window
/// would seal the graph while those errors stay queued for an owner the ended
/// window no longer names: a graph silently missing whatever they rejected.
#[test]
fn wgpu_graph_capture_is_ended_by_the_stream_that_opened_it() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();
    client.start_capture().expect("start_capture");
    launch(&client);

    // A neighbour may share the pooled stream, but not the window on it.
    let neighbour = {
        let client = client.clone();
        std::thread::spawn(move || client.stop_capture().is_err())
    };
    assert!(
        neighbour.join().unwrap(),
        "a stream that did not open the window must not close it"
    );

    // The window is untouched, so the stream that opened it still closes it.
    let graph = client.stop_capture().expect("stop_capture");
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// A graph captured on one stream keeps replaying correctly while a second
/// stream is busy — the property `requires_isolation` defends.
///
/// The scheduler may interleave two streams' tasks onto whichever stream it
/// picks first. Do that across a capture window and the recording ends up
/// holding another stream's launches, or this stream's launches execute
/// somewhere the recording never sees; either way the graph is wrong rather
/// than merely slow.
#[test]
fn wgpu_graph_capture_is_isolated_from_another_stream() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = WgpuRuntime::client(&Default::default());
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<WgpuRuntime>| {
        add_one::launch::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    client.graph_prepare().expect("graph_prepare");
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    client.start_capture().expect("start_capture");
    launch(&client);

    // A second stream runs its own unrelated work across the window.
    let other = {
        let other_client = client.clone();
        std::thread::spawn(move || {
            let a = other_client.create_from_slice(f32::as_bytes(&[7.0, 7.0, 7.0, 7.0]));
            let b = other_client.empty(n * core::mem::size_of::<f32>());
            mul_two::launch::<WgpuRuntime>(
                &other_client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(&other_client, n),
                unsafe { BufferArg::from_raw_parts(a.clone(), n) },
                unsafe { BufferArg::from_raw_parts(b.clone(), n) },
            );
            let bytes = other_client.read_one(b).unwrap();
            f32::from_bytes(&bytes).to_vec()
        })
    };
    let other = other.join().expect("the other stream's work");
    assert_eq!(
        other,
        &[14.0, 14.0, 14.0, 14.0],
        "the other stream's result"
    );

    let graph = client.stop_capture().expect("stop_capture");

    // The recorded pass is exactly this stream's one launch: replaying adds one.
    unsafe { graph.replay() };
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

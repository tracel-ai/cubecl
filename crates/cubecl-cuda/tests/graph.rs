//! Validates CUDA graph capture/replay on the actual device.

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_cuda::CudaRuntime;
use std::sync::Mutex;

/// Graph capture toggles device-global allocation state (persistent mode) on
/// the one cached client, and `CU_STREAM_CAPTURE_MODE_GLOBAL` makes any
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

    let launch = |client: &ComputeClient<CudaRuntime>| {
        add_one::launch::<CudaRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Prepare arms the persistent pool; it is mandatory before a capture.
    client.graph_prepare().expect("graph_prepare");

    // Warm up: compile the kernel and allocate every buffer, so capture stays
    // on the warm path (no compile / alloc / sync mid-capture).
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

/// A capture window that has to grow the memory pool must be REJECTED, not handed back.
///
/// A stream-ordered allocation (`cuMemAllocAsync`) issued while recording is captured as a memory
/// node, and a graph holding an allocation node it never frees cannot be relaunched: the first
/// `cuGraphLaunch` succeeds and every later one fails with `CUDA_ERROR_INVALID_VALUE`. Nothing
/// else catches this — instantiation succeeds and `cuGraphUpload` returns `CUDA_SUCCESS` — so a
/// caller that trusted `stop_capture` would only discover it on its second replay, far from the
/// cause. Warmup usually leaves the persistent pool able to serve the recorded run, so real
/// workloads hit the growth path only intermittently; this forces it by allocating a size the pool
/// has never seen inside the window.
#[test]
fn cuda_graph_capture_growing_the_pool_is_rejected() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<CudaRuntime>| {
        add_one::launch::<CudaRuntime>(
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
    // Force the pool to grow mid-capture: this deliberately odd size has never been allocated on
    // this client, so no free slice fits it and the storage must reach the device allocator.
    let grown = client.empty(3_145_733);
    let rejected = client.stop_capture();

    assert!(
        rejected.is_err(),
        "a capture that grew the pool recorded a memory node and is not relaunchable, so \
         stop_capture must reject it rather than return a graph that fails on its second replay"
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

    let launch = |client: &ComputeClient<CudaRuntime>| {
        add_one::launch::<CudaRuntime>(
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

    // Write new inputs into the captured buffer (same pointer), replay: the
    // output must reflect the new input, not the captured-time values.
    client.write(
        &input,
        Bytes::from_bytes_vec(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]).to_vec()),
    );
    unsafe { graph.replay() };
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
/// before the second reads it — write-before-read), and with buffer retention
/// (`graph_prepare` routes capture-phase allocations into the persistent pool,
/// warmup populates it, `end_capture` pins those slices) a later allocation
/// can no longer reuse `tmp`'s slice, so replay does **not** clobber the
/// sentinels. This is the acceptance test for that retention.
#[test]
fn cuda_graph_intermediate_recycling() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = CudaRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(bytes);

    let run = |client: &ComputeClient<CudaRuntime>, tmp: &Handle| {
        add_one::launch::<CudaRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
        );
        mul_two::launch::<CudaRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Prepare: capture-phase allocations now go to the persistent pool and are
    // snapshotted for retention.
    client.graph_prepare().expect("graph_prepare");

    // Warm up so `tmp` is compiled + allocated in the persistent pool (then
    // freed, so the capture run reuses it without a fresh malloc).
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

    // Drop `tmp` — the graph still references its pointer, but the pool now
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
/// drop-queue flush threshold of 64 pushes, so the deferred-flush/pool-priming
/// path is exercised) of a `Tensor` kernel that reads `shape(0)`, forcing
/// every launch through the dynamic-metadata staging + info-cache path. Then
/// verify the recorded pass did not execute, two replays re-run it exactly,
/// and an in-place input rewrite feeds the next replay.
#[test]
fn cuda_graph_many_launches_dynamic_metadata() {
    const N: usize = 64; // elements per tensor; one 64-thread block covers them
    const PASS_LAUNCHES: usize = 150; // > 2x the drop-queue flush threshold

    // One pass: ping-pong `dst = src + 1` between `a` and `b`. The identical
    // sequence is run once as warmup and once recorded.
    fn run_pass(client: &ComputeClient<CudaRuntime>, a: &Handle, b: &Handle) {
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

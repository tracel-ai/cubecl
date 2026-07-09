//! Validates HIP graph capture/replay on the actual device.

use cubecl_common::bytes::Bytes;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::Handle;
use cubecl_hip::HipRuntime;
use std::sync::Mutex;

/// Graph capture toggles device-global allocation state (persistent mode) on
/// the one cached client, so two captures must not overlap — exactly one
/// capture at a time per device, as in real use. Serialize the tests instead
/// of relying on `--test-threads 1`.
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
fn hip_graph_capture_replay() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = HipRuntime::client(&Default::default());

    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<HipRuntime>| {
        add_one::launch::<HipRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };

    // Warm up: compile the kernel and allocate every buffer, so capture stays
    // on the warm path (no compile / alloc / sync mid-capture).
    launch(&client);
    let _ = client.read_one(output.clone()).unwrap();

    // Record one launch into a graph instead of executing it.
    client.start_capture().expect("start_capture");
    launch(&client);
    let graph = client.stop_capture().expect("stop_capture");

    // Replay executes the recorded launch; the output is input + 1.
    graph.replay().expect("replay");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Replaying again re-runs it deterministically.
    graph.replay().expect("replay 2");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

/// The input-rewrite path: a captured graph reads its input buffer at replay
/// time, so writing new bytes into that same buffer (same device pointer) and
/// replaying must produce output for the new input. This is how a decode loop
/// feeds the next token into a captured step without re-capturing.
#[test]
fn hip_graph_input_rewrite() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = HipRuntime::client(&Default::default());
    let n = 4usize;
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(n * core::mem::size_of::<f32>());

    let launch = |client: &ComputeClient<HipRuntime>| {
        add_one::launch::<HipRuntime>(
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

    graph.replay().expect("replay");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Write new inputs into the captured buffer (same pointer), replay: the
    // output must reflect the new input, not the captured-time values.
    client
        .write(
            &input,
            Bytes::from_bytes_vec(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]).to_vec()),
        )
        .expect("write");
    graph.replay().expect("replay after rewrite");
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
/// Validated on gfx1151: the graph's own output stays correct (its first
/// kernel rewrites `tmp` before the second reads it — write-before-read), and
/// with buffer retention (approach B: `graph_prepare` routes capture-phase
/// allocations into the persistent pool, warmup populates it, `end_capture`
/// pins those slices) a later allocation can no longer reuse `tmp`'s slice, so
/// replay does **not** clobber the sentinels. This is the acceptance test for
/// that fix.
#[test]
fn hip_graph_intermediate_recycling() {
    let _guard = CAPTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let client = HipRuntime::client(&Default::default());
    let n = 4usize;
    let bytes = n * core::mem::size_of::<f32>();
    let input = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let output = client.empty(bytes);

    let run = |client: &ComputeClient<HipRuntime>, tmp: &Handle| {
        add_one::launch::<HipRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(tmp.clone(), n) },
        );
        mul_two::launch::<HipRuntime>(
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
    graph.replay().expect("replay");
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

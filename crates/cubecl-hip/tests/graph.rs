//! Validates HIP graph capture/replay on the actual device.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_hip::HipRuntime;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// Capture a single kernel launch into a graph, replay it, and check the
/// output — the end-to-end proof that hardware graph capture works on this GPU.
#[test]
fn hip_graph_capture_replay() {
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
    graph.replay(&client).expect("replay");
    let out = client.read_one(output.clone()).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);

    // Replaying again re-runs it deterministically.
    graph.replay(&client).expect("replay 2");
    let out = client.read_one(output).unwrap();
    assert_eq!(f32::from_bytes(&out), &[2.0, 3.0, 4.0, 5.0]);
}

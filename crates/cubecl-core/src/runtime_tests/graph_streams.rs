//! A graph recorded on one stream that reads memory another stream of its device made.
//!
//! Not part of `testgen_all`: a capture holds its stream for the whole window, so a runtime calls
//! these from its graph tests, one capture at a time.

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;
use std::time::{Duration, Instant};

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

const ELEMENTS: usize = 4;
const LAUNCHES: usize = 10;

/// A graph keeps the memory of another stream its launches read for as long as it lives, and
/// lets it go when it drops: a replay reads it again, and nothing of it is left bound after.
pub fn a_graph_holds_another_streams_input_until_it_drops<R: Runtime>() {
    let client = R::client(&Default::default());
    // Planned on an input of its own, made after the recorded one, so only the recording binds
    // `input` and the recorded run needs no wait on its stream.
    let (input, planning_input) = StreamId { value: 3_200_001 }.executes(|| {
        (
            client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0])),
            client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0])),
        )
    });

    StreamId { value: 3_200_002 }.executes(|| {
        let output = client.empty(ELEMENTS * core::mem::size_of::<f32>());
        let run = |from: &Handle| {
            for _ in 0..LAUNCHES {
                add_one::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(ELEMENTS as u32),
                    unsafe { BufferArg::from_raw_parts(from.clone(), ELEMENTS) },
                    unsafe { BufferArg::from_raw_parts(output.clone(), ELEMENTS) },
                );
            }
        };
        client.graph_prepare().unwrap();
        run(&planning_input);
        client.graph_end_run().unwrap();
        client.start_capture().unwrap();
        run(&input);
        let graph = client.stop_capture().expect("the recording seals");

        unsafe { graph.replay() }.expect("replay enqueues");
        let read = client.read_one(output.clone()).unwrap();
        assert_eq!(f32::from_bytes(&read), &[2.0, 3.0, 4.0, 5.0]);
        assert!(input.memory.is_bound(), "the graph holds the input");

        drop(graph);
        cubecl_environment::future::block_on(client.sync()).unwrap();
    });

    let start = Instant::now();
    while input.memory.is_bound() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the input should be released once the graph is gone"
        );
        std::thread::yield_now();
    }
}

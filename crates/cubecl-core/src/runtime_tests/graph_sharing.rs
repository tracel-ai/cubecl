//! Graphs recorded on one stream share the pages their captures pinned: a later capture's
//! temporaries are carved from the room earlier graphs' temporaries left free, so the memory
//! the recordings hold is the largest graph's rather than the sum of all of them, while every
//! buffer a handle still holds keeps its own memory, and every output a captured work leaves
//! allocated sits on room of its own.
//!
//! Not part of `testgen_all`: a capture holds its stream for the whole window, so a runtime calls
//! these from its graph tests, one capture at a time.

use crate::{self as cubecl};
use alloc::vec;
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::client::Graph;
use cubecl_runtime::memory_management::MemoryScope;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;
use std::println;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// Elements of every buffer: 8 MiB of `f32`, well past what the small-allocation pool serves,
/// so each temporary sits on the pages a capture pins.
const ELEMENTS: usize = 2 * 1024 * 1024;
const BYTES: usize = ELEMENTS * core::mem::size_of::<f32>();
const CUBE: u32 = 256;

/// `output = input + temporaries + 1`, through `temporaries` buffers allocated by the run and
/// all alive until its last launch, then dropped: the graph's working set is that many
/// buffers, and none of them outlives the run.
fn run_chain(client: &Client, input: &Handle, output: &Handle, temporaries: usize) {
    let launch = |from: &Handle, into: &Handle| {
        add_one::launch(
            client,
            CubeCount::Static(ELEMENTS as u32 / CUBE, 1, 1),
            CubeDim::new_1d(CUBE),
            unsafe { BufferArg::from_raw_parts(from.clone(), ELEMENTS) },
            unsafe { BufferArg::from_raw_parts(into.clone(), ELEMENTS) },
        );
    };
    let held: Vec<Handle> = (0..temporaries).map(|_| client.empty(BYTES)).collect();
    let mut from = input;
    for temporary in &held {
        launch(from, temporary);
        from = temporary;
    }
    launch(from, output);
    drop(held);
}

/// Record [`run_chain`] as a graph.
fn record_chain(client: &Client, input: &Handle, output: &Handle, temporaries: usize) -> Graph {
    let (graph, ()) = client
        .capture(|| run_chain(client, input, output, temporaries))
        .expect("capture");
    graph
}

/// Record [`run_chain`] into an output the work itself allocates, as a model's step returns its
/// logits: the graph, and the output its recorded run allocated.
fn record_chain_into_new(client: &Client, input: &Handle, temporaries: usize) -> (Graph, Handle) {
    client
        .capture(|| {
            let output = client.empty(BYTES);
            run_chain(client, input, &output, temporaries);
            output
        })
        .expect("capture")
}

/// The bytes the client's stream holds in device pages, every pool together: the host
/// staging the reads use is left out.
fn reserved(client: &Client) -> u64 {
    client
        .memory_report(MemoryScope::CurrentStream)
        .streams
        .iter()
        .map(|stream| stream.pools.usage().bytes_reserved)
        .sum()
}

fn filled(client: &Client, value: f32) -> Handle {
    client.create_from_slice(f32::as_bytes(&vec![value; ELEMENTS]))
}

fn bytes(value: f32) -> Bytes {
    Bytes::from_bytes_vec(f32::as_bytes(&vec![value; ELEMENTS]).to_vec())
}

fn read(client: &Client, handle: &Handle) -> Vec<f32> {
    f32::from_bytes(&client.read_one(handle.clone()).unwrap()).to_vec()
}

/// A smaller graph recorded after a larger one on the same stream fits in the pages the larger
/// one pinned: the stream holds no more memory after the second recording than after the
/// first, where pinning per graph would have added the second graph's working set.
pub fn a_smaller_graph_fits_in_the_pages_a_larger_one_pinned<R: Runtime>() {
    let client = R::client(&Default::default());
    StreamId { value: 3_100_001 }.executes(|| {
        let (input, wide_out, narrow_out) = (
            filled(&client, 0.0),
            client.empty(BYTES),
            client.empty(BYTES),
        );
        let before = reserved(&client);

        let wide = record_chain(&client, &input, &wide_out, 4);
        let after_wide = reserved(&client);
        let narrow = record_chain(&client, &input, &narrow_out, 2);
        let after_narrow = reserved(&client);

        println!(
            "reserved: before {before} B, after the 4-temporary graph {after_wide} B, \
             after the 2-temporary graph {after_narrow} B"
        );
        assert!(
            after_wide >= before + 4 * BYTES as u64,
            "the wide graph's temporaries took pages of their own"
        );
        assert_eq!(
            after_narrow, after_wide,
            "the narrow graph's temporaries were carved from the room the wide one pinned"
        );

        unsafe { narrow.replay() }.expect("replay enqueues");
        unsafe { wide.replay() }.expect("replay enqueues");
        assert_eq!(read(&client, &wide_out), vec![5.0; ELEMENTS]);
        assert_eq!(read(&client, &narrow_out), vec![3.0; ELEMENTS]);
    });
}

/// Graphs sharing pinned pages replay correctly in any order, each over the inputs written
/// before it: a replay overwrites only the room temporaries use, which no other graph needs
/// between its own replays.
pub fn graphs_sharing_pinned_pages_replay_in_any_order<R: Runtime>() {
    let client = R::client(&Default::default());
    StreamId { value: 3_100_002 }.executes(|| {
        let (wide_in, narrow_in) = (filled(&client, 0.0), filled(&client, 0.0));
        let (wide_out, narrow_out) = (client.empty(BYTES), client.empty(BYTES));
        let wide = record_chain(&client, &wide_in, &wide_out, 3);
        let narrow = record_chain(&client, &narrow_in, &narrow_out, 1);

        for (step, replay_wide_first) in [false, true, true, false].into_iter().enumerate() {
            let base = (step * 10) as f32;
            client.write(&wide_in, bytes(base));
            client.write(&narrow_in, bytes(base + 1.0));
            let order = match replay_wide_first {
                true => [&wide, &narrow],
                false => [&narrow, &wide],
            };
            for graph in order {
                unsafe { graph.replay() }.expect("replay enqueues");
            }
            assert_eq!(
                read(&client, &wide_out),
                vec![base + 4.0; ELEMENTS],
                "step {step}"
            );
            assert_eq!(
                read(&client, &narrow_out),
                vec![base + 3.0; ELEMENTS],
                "step {step}"
            );
        }
    });
}

/// An output the captured work allocated keeps its own memory: a graph recorded after it on the
/// same stream never carves it, so it survives that graph's capture and replays, and the host
/// reads what its own graph wrote.
pub fn a_held_output_survives_another_graphs_replay<R: Runtime>() {
    let client = R::client(&Default::default());
    StreamId { value: 3_100_003 }.executes(|| {
        let input = filled(&client, 0.0);
        let (first, first_out) = record_chain_into_new(&client, &input, 2);

        unsafe { first.replay() }.expect("replay enqueues");
        assert_eq!(read(&client, &first_out), vec![3.0; ELEMENTS]);

        // A second graph whose temporaries need every byte the first one left free, and more.
        let other_in = filled(&client, 100.0);
        let other_out = client.empty(BYTES);
        let second = record_chain(&client, &other_in, &other_out, 4);
        unsafe { second.replay() }.expect("replay enqueues");

        assert_eq!(
            read(&client, &first_out),
            vec![3.0; ELEMENTS],
            "the second graph's capture and replay left the first graph's output alone"
        );
        assert_eq!(read(&client, &other_out), vec![105.0; ELEMENTS]);
    });
}

/// An output a later capture's work allocated never sits on the room an earlier graph's
/// temporaries use: replaying the earlier graph after the later one leaves the later graph's
/// output as its replay wrote it, so the host may read it whenever it likes. Without the
/// planning placing it, the output is carved from the earlier graph's free room, and that
/// graph's replay writes its temporaries over it.
pub fn an_output_survives_an_earlier_graphs_replay<R: Runtime>() {
    let client = R::client(&Default::default());
    StreamId { value: 3_100_004 }.executes(|| {
        let (wide_in, wide_out) = (filled(&client, 0.0), client.empty(BYTES));
        let wide = record_chain(&client, &wide_in, &wide_out, 4);
        let narrow_in = filled(&client, 10.0);
        let (narrow, narrow_out) = record_chain_into_new(&client, &narrow_in, 1);

        unsafe { narrow.replay() }.expect("replay enqueues");
        unsafe { wide.replay() }.expect("replay enqueues");
        assert_eq!(
            read(&client, &narrow_out),
            vec![12.0; ELEMENTS],
            "the wide graph's replay left the narrow graph's output alone"
        );
        assert_eq!(read(&client, &wide_out), vec![5.0; ELEMENTS]);
    });
}

/// Elements of a small output: 1 MiB of `f32`, well under the pages the temporaries grow.
const SMALL_ELEMENTS: usize = ELEMENTS / 8;
const SMALL_BYTES: usize = SMALL_ELEMENTS * core::mem::size_of::<f32>();

/// Record a work that goes through two 8 MiB temporaries and returns a 1 MiB output it allocates
/// itself: the graph, and that output.
fn record_small_output(client: &Client, input: &Handle) -> (Graph, Handle) {
    let launch = |from: &Handle, into: &Handle, elements: usize| {
        add_one::launch(
            client,
            CubeCount::Static(ELEMENTS as u32 / CUBE, 1, 1),
            CubeDim::new_1d(CUBE),
            unsafe { BufferArg::from_raw_parts(from.clone(), ELEMENTS) },
            unsafe { BufferArg::from_raw_parts(into.clone(), elements) },
        );
    };
    client
        .capture(|| {
            let (first, second) = (client.empty(BYTES), client.empty(BYTES));
            launch(input, &first, ELEMENTS);
            launch(&first, &second, ELEMENTS);
            let output = client.empty(SMALL_BYTES);
            launch(&second, &output, SMALL_ELEMENTS);
            output
        })
        .expect("capture")
}

/// The outputs of a stream's graphs pack together, apart from the room their temporaries share:
/// each graph adds about its output, not a page of its own. Placing an output merely off the
/// pinned pages would take a fresh page per graph, since every page a recording touches is
/// pinned, and the memory sharing saves would come back one page per graph.
pub fn outputs_pack_apart_from_the_shared_room<R: Runtime>() {
    const GRAPHS: usize = 6;
    let client = R::client(&Default::default());
    StreamId { value: 3_100_006 }.executes(|| {
        let input = filled(&client, 1.0);
        let first = record_small_output(&client, &input);
        let after_first = reserved(&client);
        let rest: Vec<(Graph, Handle)> = (1..GRAPHS)
            .map(|_| record_small_output(&client, &input))
            .collect();
        let added = reserved(&client) - after_first;

        println!(
            "reserved after the first graph {after_first} B, added by {} more: {added} B",
            GRAPHS - 1
        );
        assert!(
            added <= 2 * ((GRAPHS - 1) * SMALL_BYTES) as u64,
            "{added} B for {} outputs of {SMALL_BYTES} B each",
            GRAPHS - 1
        );

        for (graph, _) in core::iter::once(&first).chain(&rest) {
            unsafe { graph.replay() }.expect("replay enqueues");
        }
        for (_, output) in core::iter::once(&first).chain(&rest) {
            assert_eq!(read(&client, output), vec![4.0; SMALL_ELEMENTS]);
        }
    });
}

/// A recorded run that keeps a buffer its planning run let go does not seal: that buffer was
/// placed as a temporary, on room another graph's replay may write.
pub fn a_recording_that_keeps_a_planned_temporary_does_not_seal<R: Runtime>() {
    let client = R::client(&Default::default());
    StreamId { value: 3_100_005 }.executes(|| {
        let input = filled(&client, 0.0);
        let run = || {
            let output = client.empty(BYTES);
            run_chain(&client, &input, &output, 1);
            output
        };
        client.graph_prepare().expect("graph_prepare");
        // Each planning run lets the buffer go before it ends: a temporary.
        while !{
            drop(run());
            client.graph_end_run().expect("graph_end_run")
        } {}
        client.start_capture().expect("start_capture");
        let kept = run();
        assert!(
            client.stop_capture().is_err(),
            "the recording kept what its planning run found to be a temporary"
        );
        drop(kept);
        assert!(!client.is_capturing(), "a refused capture is over");
    });
}

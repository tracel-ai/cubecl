//! What a client reads about its stream's graph capture: [`Client::is_capturing`] is true from
//! `graph_prepare` until the capture ends, for every client of the capturing stream and for no
//! other stream.
//!
//! Not part of `testgen_all`: a capture holds its stream for the whole window, so a runtime calls
//! these from its graph tests, one capture at a time.

use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn add_one(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// A capture is visible to every client of the stream that prepared it, through the warmup run
/// and the recorded one, and to no other stream.
///
/// Code that has to decide the same way in the warmup and the recording reads it before every
/// launch, possibly through a client other than the one that prepared, so each client of the
/// stream has to see it; a neighbour that saw it would take the capture's path for nothing.
pub fn a_capture_is_seen_by_every_client_of_its_stream<R: Runtime>() {
    let client = R::client(&Default::default());
    let other_client = R::client(&Default::default());
    let owner = StreamId { value: 3_000_001 };
    let neighbour = StreamId { value: 3_000_002 };
    let capturing = |client: &Client, stream: StreamId| stream.executes(|| client.is_capturing());

    let n = 4usize;
    let (input, output) = owner.executes(|| {
        (
            client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0])),
            client.empty(n * core::mem::size_of::<f32>()),
        )
    });
    let run = || {
        add_one::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(&client, n),
            unsafe { BufferArg::from_raw_parts(input.clone(), n) },
            unsafe { BufferArg::from_raw_parts(output.clone(), n) },
        );
    };
    owner.executes(|| {
        run();
        client.read_one(output.clone()).unwrap();
    });

    assert!(!capturing(&client, owner), "no capture before prepare");
    owner
        .executes(|| client.graph_prepare())
        .expect("graph_prepare");
    assert!(capturing(&client, owner), "the warmup run is part of it");
    assert!(
        capturing(&other_client, owner),
        "every client of the stream sees it"
    );
    assert!(
        !capturing(&client, neighbour),
        "another stream isn't capturing"
    );

    owner.executes(|| {
        run();
        client.read_one(output.clone()).unwrap();
        client.start_capture().expect("start_capture");
    });
    assert!(
        capturing(&other_client, owner),
        "the recorded run is part of it"
    );
    assert!(!capturing(&other_client, neighbour));

    let graph = owner.executes(|| {
        run();
        client.stop_capture()
    });
    graph.expect("stop_capture");
    assert!(
        !capturing(&client, owner),
        "the capture ends with stop_capture"
    );
    assert!(!capturing(&other_client, owner));
}

/// A capture prepared but never opened ends at `stop_capture`, which is the only call left to a
/// caller whose warmup failed: the stream stops reporting it, and can be prepared again.
pub fn a_prepared_capture_ends_at_stop_capture<R: Runtime>() {
    let client = R::client(&Default::default());
    let owner = StreamId { value: 3_000_003 };
    let capturing = || owner.executes(|| client.is_capturing());

    owner
        .executes(|| client.graph_prepare())
        .expect("graph_prepare");
    assert!(capturing(), "the warmup run is part of the capture");
    owner
        .executes(|| client.stop_capture())
        .expect_err("nothing recorded, so there is no graph to seal");
    assert!(!capturing(), "the capture ended with stop_capture");

    owner
        .executes(|| client.graph_prepare())
        .expect("the stream can be prepared again");
    assert!(capturing());
    let _ = owner.executes(|| client.stop_capture());
    assert!(!capturing());
}

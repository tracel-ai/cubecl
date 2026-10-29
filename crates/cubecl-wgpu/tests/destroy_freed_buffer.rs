//! The storage destroys a buffer it frees, since on the web a dropped buffer
//! frees nothing until the browser collects it. A launch still queued holds
//! the buffers it binds, so the buffer must outlive it: freed under the
//! launch, it is still read by it.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;

#[cube(launch)]
fn copy(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

#[test]
fn a_buffer_freed_under_a_queued_launch_is_still_read_by_it() {
    let client = <WgpuRuntime>::client(&Default::default());
    let output = client.empty(core::mem::size_of::<u32>());

    // A dedicated allocation goes back to the storage at the next reservation
    // after it is freed, with no cleanup of the stream's queue first.
    client.memory_dedicated_allocation(&client, |client| {
        let input = client.create_from_slice(u32::as_bytes(&[7]));
        // The write holds the input until it runs.
        client.flush().expect("the write runs");
        copy::launch(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(input.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
        );
        // The queued launch is all that still binds the input's buffer.
        drop(input);
        drop(client.empty(core::mem::size_of::<u32>()));
    });

    let bytes = client
        .read_one(output)
        .expect("the launch ran on its input");
    assert_eq!(u32::from_bytes(&bytes), &[7]);
}

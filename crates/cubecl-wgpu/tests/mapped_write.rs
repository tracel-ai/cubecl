//! A mapped write lands after the queued writes issued before it, whichever stream queued them.
//!
//! A queued write lands at the next submit on the queue the streams share, so a copy through the
//! mapping made before then is overwritten. The poll cannot see this: a write nothing submitted
//! yet leaves the queue idle.
//!
//! Vulkan only: only its integrated GPUs map their buffers, and elsewhere both writes take the
//! queue, which orders them. One test per binary: the mapped path needs an idle queue.
#![cfg(all(feature = "spirv", not(target_family = "wasm")))]

use cubecl_common::bytes::Bytes;
use cubecl_common::device::Device;
use cubecl_environment::stream::StreamId;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuDeviceKind, WgpuRuntime};

#[test]
fn a_mapped_write_lands_after_a_write_queued_on_another_stream() {
    let device = WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(0));
    if <WgpuRuntime>::find_device(device.to_id()).is_err() {
        eprintln!("no integrated GPU, so nothing here maps its buffers");
        return;
    }
    let client = <WgpuRuntime>::client(&device);

    // Two ids far enough apart to land on different pooled streams for any `max_streams`.
    let owner = StreamId { value: 2_000_001 };
    let writer = StreamId { value: 2_000_002 };

    let buf = owner.executes(|| {
        let buf = client.empty(5);
        // Longer than the buffer by its padding, which the mapping refuses whatever the queue is
        // doing, so this one takes the queue.
        client.write(&buf, Bytes::from_bytes_vec(vec![1; 8]));
        buf
    });

    // Registering this runs the owner's write, which nothing submits: the owner has no tasks.
    writer.executes(|| client.write(&buf, Bytes::from_bytes_vec(vec![2; 5])));

    let read = writer.executes(|| client.read_one(buf).expect("both writes succeeded"));
    assert_eq!(
        &read[..],
        &[2; 5],
        "the second write was issued last, so it has to land last"
    );
}

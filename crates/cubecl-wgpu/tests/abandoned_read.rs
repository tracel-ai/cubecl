//! A read dropped before it resolves hands its staging buffer back unmapped, so
//! the next read of the same size, which reuses that buffer, still works.
#![cfg(not(target_family = "wasm"))]

use cubecl_common::bytes::Bytes;
use cubecl_core::prelude::*;
use cubecl_core::{MemoryScope, MemoryUsage};
use cubecl_environment::future::block_on;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[test]
fn a_read_dropped_unawaited_releases_its_staging_buffer_unmapped() {
    let client = <WgpuRuntime>::client(&WgpuDevice::default());
    let values = vec![7u8; 256];
    let handle = client.create(Bytes::from_bytes_vec(values.clone()));

    core::mem::drop(client.read_async(vec![handle.clone()]));
    block_on(client.sync()).expect("the device runs");
    assert_eq!(staging_in_use(&client).in_use_bytes, 0);

    let read = client
        .read_one(handle)
        .expect("the staging buffer was unmapped");
    assert_eq!(&read[..], &values[..]);
}

fn staging_in_use(client: &Client) -> MemoryUsage {
    client
        .memory_report(MemoryScope::CurrentStream)
        .streams
        .iter()
        .flat_map(|stream| &stream.auxiliary)
        .filter(|memory| memory.name == "Staging CPU Memory")
        .fold(MemoryUsage::default(), |usage, memory| {
            usage.combine(memory.pools.usage())
        })
}

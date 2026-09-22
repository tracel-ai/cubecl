//! The adaptive memory on the actual device: an allocation the pages held no
//! longer fit outdates the pool holding them, and an explicit cleanup
//! relocates what is still live on it — bytes included — into the room the
//! pool carving allocations already has.

use cubecl_hip::HipRuntime;
use cubecl_server::memory_management::{MemoryPoolKind, MemoryScope};
use cubecl_server::runtime::Runtime;

const MIB: usize = 1024 * 1024;

#[test]
fn relocation_moves_live_bytes_off_outdated_pages() {
    // The adaptive memory is what every device that slices gets.
    let client = HipRuntime::client(&Default::default());
    client.memory_cleanup().expect("no stream records a graph");

    let pattern: Vec<u8> = (0..4 * MIB).map(|i| (i % 251) as u8).collect();
    let kept = client.create_from_slice(&pattern);
    // Larger than `kept`'s page: the pool holding it is outdated, and a pool
    // carving 101 MiB pages takes over.
    let large = client.empty(100 * MIB);
    assert_eq!(adaptive(&client), (101 * MIB as u64, 1));

    // The cleanup relocates before it releases anything: the page `large`
    // leaves free is the room `kept` moves into.
    drop(large);

    client.memory_cleanup().expect("no stream records a graph");

    assert_eq!(
        adaptive(&client),
        (101 * MIB as u64, 0),
        "the outdated pool is gone"
    );
    let bytes = client.read_one(kept).unwrap();
    assert_eq!(
        &bytes[..],
        &pattern[..],
        "the bytes moved with the allocation"
    );
}

/// The size the adaptive memory carves pages at, and how many pages its
/// outdated pools still hold.
fn adaptive(client: &cubecl_server::client::Client) -> (u64, u64) {
    client
        .memory_report(MemoryScope::CurrentStream)
        .streams
        .iter()
        .flat_map(|stream| stream.pools.dynamic.iter())
        .find_map(|pool| match pool.kind {
            MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            } => Some((page_size, outdated_pages)),
            _ => None,
        })
        .expect("the adaptive preset has an adaptive pool")
}

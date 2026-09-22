//! The adaptive pool on the actual device: a grown page size outdates the
//! pages already held, and an explicit cleanup relocates what is live on them —
//! bytes included — onto pages of the new size.

use cubecl_hip::HipRuntime;
use cubecl_server::config::memory::{MemoryPoolsConfig, MemoryPoolsPreset};
use cubecl_server::memory_management::MemoryPoolKind;
use cubecl_server::runtime::Runtime;

const MIB: usize = 1024 * 1024;

#[test]
fn relocation_moves_live_bytes_off_outdated_pages() {
    let client = HipRuntime::client(&Default::default());
    client.memory_cleanup();
    client
        .install_memory_pools(&MemoryPoolsConfig::Preset(MemoryPoolsPreset::Adaptive))
        .unwrap();

    let pattern: Vec<u8> = (0..4 * MIB).map(|i| (i % 251) as u8).collect();
    let kept = client.create_from_slice(&pattern);
    // Past the 64 MiB floor: the page size grows and `kept`'s page is outdated.
    let large = client.empty(100 * MIB);
    assert_eq!(adaptive(&client), (101 * MIB as u64, 1));

    client.memory_cleanup();

    assert_eq!(
        adaptive(&client),
        (101 * MIB as u64, 0),
        "the outdated page is gone"
    );
    let bytes = client.read_one(kept).unwrap();
    assert_eq!(
        &bytes[..],
        &pattern[..],
        "the bytes moved with the allocation"
    );
    drop(large);
}

/// The adaptive pool's page size and outdated page count.
fn adaptive(client: &cubecl_server::client::Client) -> (u64, u64) {
    client
        .memory_report()
        .dynamic
        .iter()
        .find_map(|pool| match pool.kind {
            MemoryPoolKind::Adaptive {
                page_size,
                outdated_pages,
            } => Some((page_size, outdated_pages)),
            _ => None,
        })
        .expect("the adaptive preset has an adaptive pool")
}

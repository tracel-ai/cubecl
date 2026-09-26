//! The adaptive memory on a runtime's device: an allocation the pages held no
//! longer fit outdates the pool holding them, and an explicit cleanup
//! relocates what is still live on it, bytes included.
//!
//! Not part of `testgen_all`: each check reads what the whole memory holds, so
//! it needs a process to itself. A runtime calls these from an integration
//! test of their own.

use alloc::vec::Vec;

use crate::{MemoryPoolKind, MemoryScope};
use cubecl_runtime::{client::Client, runtime::Runtime};

const MIB: usize = 1024 * 1024;
/// Not a multiple of any copy alignment: a relocation copies the rounded size,
/// which has to stay inside both slices.
const KEPT_SIZE: usize = 4 * MIB + 3;

/// A live allocation on an outdated page moves onto the current pages at an
/// explicit cleanup, its bytes with it, and the outdated pool goes back.
pub fn relocation_moves_live_bytes_off_outdated_pages<R: Runtime>() {
    let client = R::client(&Default::default());
    client.memory_cleanup().expect("no stream records a graph");

    let pattern = pattern(KEPT_SIZE);
    let kept = client.create_from_slice(&pattern);
    let (page_size, _) = adaptive(&client);
    // Larger than `kept`'s page: the pool holding it is outdated, and a pool
    // carving larger pages takes over.
    let large = client.empty(page_size as usize + MIB);
    assert_eq!(adaptive(&client).1, 1, "`kept`'s page is outdated");

    // The cleanup relocates before it releases anything: the page `large`
    // leaves free is the room `kept` moves into.
    drop(large);
    client.memory_cleanup().expect("no stream records a graph");

    assert_eq!(adaptive(&client).1, 0, "the outdated pool is gone");
    let bytes = client.read_one(kept).unwrap();
    assert_eq!(bytes.len(), KEPT_SIZE, "the read-back ran");
    assert_eq!(
        &bytes[..],
        &pattern[..],
        "the bytes moved with the allocation"
    );
}

/// On a runtime whose read-backs point into the memory they read, a held
/// read-back keeps its allocation where it is: a cleanup leaves its page
/// outdated rather than freeing the bytes the read-back points at.
pub fn a_held_read_back_keeps_its_allocation_in_place<R: Runtime>() {
    let client = R::client(&Default::default());
    client.memory_cleanup().expect("no stream records a graph");

    let pattern = pattern(KEPT_SIZE);
    let kept = client.create_from_slice(&pattern);
    let read = client.read_one(kept.clone()).unwrap();
    let (page_size, _) = adaptive(&client);
    let large = client.empty(page_size as usize + MIB);
    drop(large);

    client.memory_cleanup().expect("no stream records a graph");
    assert_eq!(
        adaptive(&client).1,
        1,
        "the read-back's page stays where it is"
    );
    assert_eq!(
        &read[..],
        &pattern[..],
        "the read-back still reads the bytes"
    );

    drop(read);
    // That cleanup gave the empty current page back, and an explicit
    // relocation only moves into room already held: take a current page
    // again, and leave it free for `kept`.
    drop(client.empty(KEPT_SIZE));
    client.memory_cleanup().expect("no stream records a graph");
    assert_eq!(
        adaptive(&client).1,
        0,
        "once the read-back goes, the allocation moves"
    );
    let bytes = client.read_one(kept).unwrap();
    assert_eq!(&bytes[..], &pattern[..], "the bytes moved with it");
}

fn pattern(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 251) as u8).collect()
}

/// The size the adaptive memory carves pages at, and how many pages its
/// outdated pools still hold.
fn adaptive(client: &Client) -> (u64, u64) {
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
        .expect("the default memory has an adaptive pool")
}

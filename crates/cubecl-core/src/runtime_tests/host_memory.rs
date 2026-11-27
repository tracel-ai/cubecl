//! Kernels reading host memory a device imported, in place.

use crate::MemoryScope;
use crate::{self as cubecl};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use cubecl_runtime::server::{Handle, IoError, ServerError};
use cubecl_runtime::storage::HOST_MEMORY_ALIGNMENT;
use std::println;

const WORDS: usize = HOST_MEMORY_ALIGNMENT / 4;

#[cube(launch_unchecked)]
pub fn kernel_copy_words(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

fn host_words(chunks: usize) -> Arc<Bytes> {
    let size = chunks * HOST_MEMORY_ALIGNMENT;
    let mut memory = Bytes::zeroed_aligned(size, HOST_MEMORY_ALIGNMENT).unwrap();
    for (index, word) in memory.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *word = (index as u32 ^ 0x9e37_79b9).to_le_bytes();
    }
    Arc::new(memory)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

fn copy_out(client: &Client, input: Handle) -> Vec<u32> {
    let output = client.empty(WORDS * 4);
    copy(client, input, output.clone());
    words(&client.read_one_unchecked(output))
}

fn copy(client: &Client, input: Handle, output: Handle) {
    unsafe {
        kernel_copy_words::launch_unchecked(
            client,
            CubeCount::Static((WORDS as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(input, WORDS),
            BufferArg::from_raw_parts(output, WORDS),
        )
    };
}

fn skips(client: &Client) -> bool {
    let skips = !client.features().host_memory_import;
    if skips {
        println!("Host memory import not supported, skipping");
    }
    skips
}

fn refused<T>(result: &Result<T, ServerError>) -> bool {
    matches!(
        result,
        Err(ServerError::Io(IoError::HostMemoryRefused { .. }))
    )
}

/// Two ranges of one host memory, each imported on its own, read back what the host wrote there.
pub fn test_kernel_reads_imported_ranges(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(3);
    let chunk = HOST_MEMORY_ALIGNMENT as u64;
    let first = client.import_host_memory(&memory, 0..chunk).unwrap();
    let third = client
        .import_host_memory(&memory, 2 * chunk..3 * chunk)
        .unwrap();

    let expected = words(&memory[..]);
    assert_eq!(copy_out(&client, first), expected[..WORDS]);
    assert_eq!(copy_out(&client, third), expected[2 * WORDS..]);
}

/// Queues the release of the imports dropped on `client`'s stream, then lets go of them once done.
fn release(client: &Client) {
    for _ in 0..2 {
        cubecl_environment::future::block_on(client.sync()).unwrap();
        client.memory_cleanup().unwrap();
    }
}

/// Every device of `clients` imports one host memory and reads it at once. One letting go of its
/// import while another reads leaves the memory pinned for the rest, and once the last lets go,
/// it is unpinned: the devices hold it no more and it pins again.
pub fn test_devices_read_one_import(clients: &[Client]) {
    let clients: Vec<Client> = clients
        .iter()
        .filter(|client| client.features().host_memory_import)
        .cloned()
        .collect();
    if clients.len() < 2 {
        println!("Fewer than two devices import host memory, skipping");
        return;
    }
    let mut memory = host_words(1);
    let range = 0..HOST_MEMORY_ALIGNMENT as u64;
    let expected = words(&memory[..]);
    let mut imports: Vec<Handle> = clients
        .iter()
        .map(|client| client.import_host_memory(&memory, range.clone()).unwrap())
        .collect();
    let outputs: Vec<Handle> = clients
        .iter()
        .zip(&imports)
        .map(|(client, input)| {
            let output = client.empty(WORDS * 4);
            copy(client, input.clone(), output.clone());
            output
        })
        .collect();
    for (client, output) in clients.iter().zip(outputs) {
        assert_eq!(words(&client.read_one_unchecked(output)), expected);
    }

    let reading = clients[1].empty(WORDS * 4);
    copy(&clients[1], imports[1].clone(), reading.clone());
    drop(imports.remove(0));
    release(&clients[0]);
    assert_eq!(words(&clients[1].read_one_unchecked(reading)), expected);
    assert!(Arc::get_mut(&mut memory).is_none());
    for (client, input) in clients[1..].iter().zip(&imports) {
        assert_eq!(copy_out(client, input.clone()), expected);
    }

    drop(imports);
    clients.iter().for_each(release);
    assert!(Arc::get_mut(&mut memory).is_some());
    let again = clients[0].import_host_memory(&memory, range).unwrap();
    assert_eq!(copy_out(&clients[0], again), expected);
}

/// The stream's memory report counts every import and its bytes in the imported pool.
pub fn test_imports_show_in_memory_report(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(2);
    let chunk = HOST_MEMORY_ALIGNMENT as u64;
    let _first = client.import_host_memory(&memory, 0..chunk).unwrap();
    let _second = client
        .import_host_memory(&memory, chunk..2 * chunk)
        .unwrap();

    // Other tests may hold imports on the same stream while this one runs.
    let report = client.memory_report(MemoryScope::CurrentStream);
    let imported = &report.streams[0].pools.imported.usage;
    assert!(imported.allocation_count >= 2);
    assert!(imported.in_use_bytes >= 2 * chunk);
}

/// Once its handle is dropped and the work reading it is done, a cleanup lets go of the memory.
pub fn test_cleanup_releases_imported_memory(client: Client) {
    if skips(&client) {
        return;
    }
    let mut memory = host_words(1);
    let handle = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    copy_out(&client, handle);

    // The first cleanup queues the release behind the stream, the second finds it done.
    for _ in 0..2 {
        cubecl_environment::future::block_on(client.sync()).unwrap();
        client.memory_cleanup().unwrap();
    }
    assert!(Arc::get_mut(&mut memory).is_some());
}

/// An imported handle is never mutable in place.
pub fn test_imported_handle_is_not_mutable(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(1);
    let handle = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    assert!(!handle.can_mut());
}

/// An imported handle is never shared with peers: each device imports the memory on its own.
pub fn test_imported_memory_is_not_shared(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(1);
    let handle = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    assert!(client.share_with_peers(&handle).is_err());
}

/// A write to imported memory fails its handle without reaching the host.
pub fn test_write_to_imported_memory_fails(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(1);
    let handle = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    client.write(
        &handle,
        Bytes::from_bytes_vec(vec![0xff; HOST_MEMORY_ALIGNMENT]),
    );
    assert!(client.read_one(handle).is_err());
    assert_eq!(words(&memory[..])[0], 0x9e37_79b9);
}

/// A copy of imported memory that a refused write failed carries the failure to the copy.
pub fn test_copy_of_failed_imported_memory_fails(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(1);
    let handle = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    client.write(
        &handle,
        Bytes::from_bytes_vec(vec![0xff; HOST_MEMORY_ALIGNMENT]),
    );
    let copy = client.clone().to_client(handle, &client);
    assert!(client.read_one(copy).is_err());
}

/// A kernel given imported memory as its output fails the handle without reaching the host.
pub fn test_kernel_output_to_imported_memory_fails(client: Client) {
    if skips(&client) {
        return;
    }
    let memory = host_words(1);
    let output = client
        .import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
        .unwrap();
    copy(&client, client.empty(WORDS * 4), output.clone());
    assert!(client.read_one(output).is_err());
    assert_eq!(words(&memory[..])[0], 0x9e37_79b9);
}

/// An import outside the memory, unaligned or empty is refused before it reaches the device.
pub fn test_import_refuses_bad_ranges(client: Client) {
    let memory = host_words(1);
    let chunk = HOST_MEMORY_ALIGNMENT as u64;
    for range in [0..2 * chunk, 64..chunk, 0..64, chunk..chunk] {
        let result = client.import_host_memory(&memory, range.clone());
        assert!(refused(&result), "{range:?} was not refused: {result:?}");
    }
}

/// A device lacking the Feature refuses every import.
pub fn test_import_without_feature_is_refused(client: Client) {
    if client.features().host_memory_import {
        println!("Host memory import supported, skipping");
        return;
    }
    let memory = host_words(1);
    assert!(refused(
        &client.import_host_memory(&memory, 0..HOST_MEMORY_ALIGNMENT as u64)
    ));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_host_memory {
    () => {
        mod host_memory {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_kernel_reads_imported_ranges() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_kernel_reads_imported_ranges(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_imports_show_in_memory_report() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_imports_show_in_memory_report(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_cleanup_releases_imported_memory() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_cleanup_releases_imported_memory(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_imported_handle_is_not_mutable() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_imported_handle_is_not_mutable(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_imported_memory_is_not_shared() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_imported_memory_is_not_shared(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_write_to_imported_memory_fails() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_write_to_imported_memory_fails(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_copy_of_failed_imported_memory_fails() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_copy_of_failed_imported_memory_fails(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_kernel_output_to_imported_memory_fails() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_kernel_output_to_imported_memory_fails(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_import_refuses_bad_ranges() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_import_refuses_bad_ranges(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_import_without_feature_is_refused() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::host_memory::test_import_without_feature_is_refused(client);
            }
        }
    };
}

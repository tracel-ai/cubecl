use crate::{self as cubecl};
use alloc::vec;
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_runtime::server::{MemoryLayoutDescriptor, MemoryLayoutStrategy};

const MB: usize = 1024 * 1024;

/// Past the size a backend stages whole, with a tail that ends mid-chunk.
const LARGE_BYTES: usize = 117 * MB + 3;

/// Bytes whose pattern repeats at no chunk boundary, so a chunk copied to the
/// wrong place reads back wrong.
fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| ((i + seed) % 251) as u8).collect()
}

pub fn test_large_upload_round_trips(client: Client) {
    let data = pattern(LARGE_BYTES, 0);

    let handle = client.create_from_slice(&data);
    let read = client.read_one_unchecked(handle);

    assert_eq!(read.len(), data.len());
    assert!(
        read[..] == data[..],
        "the upload should read back as written"
    );
}

pub fn test_packed_large_uploads_round_trip(client: Client) {
    let first = pattern(LARGE_BYTES, 0);
    let second = pattern(LARGE_BYTES, 7);
    let descriptor =
        || MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [LARGE_BYTES].into(), 1);

    let layouts = client.create_tensors_from_slices(vec![
        (descriptor(), first.as_slice()),
        (descriptor(), second.as_slice()),
    ]);

    for (layout, data) in layouts.into_iter().zip([first, second]) {
        let read = client.read_one_unchecked(layout.memory);
        assert_eq!(read.len(), data.len());
        assert!(
            read[..] == data[..],
            "each packed upload should read back as written"
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_upload {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_large_upload_round_trips() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::upload::test_large_upload_round_trips(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_packed_large_uploads_round_trip() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::upload::test_packed_large_uploads_round_trip(client);
        }
    };
}

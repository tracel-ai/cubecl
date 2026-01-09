use crate::{self as cubecl};
use cubecl::prelude::*;
use cubecl_common::bytes::Bytes;
use std::io::Write;

const MB: usize = 1024 * 1024;
pub fn test_file_memory<R: Runtime>(client: ComputeClient<R>) {
    let dir = tempfile::tempdir().unwrap();
    let file_name = dir.path().join("test");
    let data_init = (0i32..MB as i32).collect::<Vec<i32>>();
    let bytes_generated = i32::as_bytes(&data_init);

    let size = bytes_generated.len() as u64;
    let offset = 0;

    let mut file = std::fs::File::create(&file_name).unwrap();
    file.write_all(bytes_generated).unwrap();
    core::mem::drop(file);

    let bytes = Bytes::from_file(&file_name, size, offset);

    let bytes_from_file: &[u8] = &bytes;
    assert_eq!(
        bytes_generated, bytes_from_file,
        "The file is read correctly."
    );
    core::mem::drop(bytes);

    let bytes = Bytes::from_file(&file_name, size, offset);

    let handle = client.create(bytes);
    let bytes = client.read_one(handle);
    let bytes_from_client: &[u8] = &bytes;

    core::mem::drop(dir);
    assert_eq!(
        bytes_generated, bytes_from_client,
        "The data is correctly loaded."
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_file {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_kernel_load_file() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::file::test_file_memory::<TestRuntime>(client);
        }
    };
}

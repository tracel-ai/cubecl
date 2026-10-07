//! Kernels a compile-only launch queues: they compile together when the queue
//! does, on as many threads as the server compiles on, and then run like any
//! other kernel. The launches that queued them do not run.

use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl::prelude::*;
use cubecl_runtime::execution::CompileOnly;
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;

/// Writes a value only this instance of the kernel produces, so each output
/// says which kernel ran.
#[cube(launch)]
pub fn kernel_numbered(out: &mut [u32], #[comptime] number: u32) {
    if ABSOLUTE_POS == 0 {
        out[0] = number * 3 + 1;
    }
}

/// More kernels than one thread compiles at a time, so a server compiling on
/// several threads does.
const KERNELS: u32 = 12;

pub fn test_compiled_kernels_run<R: Runtime>(client: Client) {
    let outputs: Vec<_> = (0..KERNELS)
        .map(|_| client.create_from_slice(u32::as_bytes(&[0])))
        .collect();

    {
        let _compile_only = CompileOnly::new();
        for (number, out) in outputs.iter().enumerate() {
            launch_numbered(&client, out, number as u32);
        }
    }
    // A flush compiles nothing: only a launch that executes compiles the queue.
    client.flush().unwrap();

    for out in &outputs {
        let actual = client.read_one(out.clone()).unwrap();
        assert_eq!(u32::from_bytes(&actual), &[0], "a compile-only launch ran");
    }

    for (number, out) in outputs.iter().enumerate() {
        launch_numbered(&client, out, number as u32);
    }
    for (number, out) in outputs.into_iter().enumerate() {
        let actual = client.read_one(out).unwrap();
        assert_eq!(u32::from_bytes(&actual), &[number as u32 * 3 + 1]);
    }
}

fn launch_numbered(client: &Client, out: &Handle, number: u32) {
    kernel_numbered::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(out.clone(), 1) },
        number,
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_compile {
    () => {
        #[test]
        fn test_compiled_kernels_run() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::compile::test_compiled_kernels_run::<TestRuntime>(client);
        }
    };
}

//! The runtime's context is current for every command, whatever code outside
//! the runtime left current on its thread.

use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_environment::bytes::Bytes;
use cubecl_server::client::Client;
use cubecl_server::runtime::Runtime;

#[test]
fn commands_restore_the_runtime_context() {
    let client = CudaRuntime::client(&CudaDevice::default());
    let runtime_context = client.exclusive(current_context).unwrap();

    client.exclusive(clear_current_context).unwrap();
    assert_commands_run_in(&client, runtime_context);
}

/// Runs two commands, checking after each that `context` is current.
fn assert_commands_run_in(client: &Client, context: usize) {
    for _ in 0..2 {
        let values = vec![1_u8, 7, 19, 255];
        let handle = client.create(Bytes::from_bytes_vec(values.clone()));
        let actual = client.read_one(handle).unwrap();
        assert_eq!(&actual[..], values.as_slice());
        assert_eq!(client.exclusive(current_context).unwrap(), context);
    }
}

/// The calling thread's current context, as an address that can cross threads.
fn current_context() -> usize {
    cudarc::driver::result::ctx::get_current().unwrap().unwrap() as usize
}

/// Leaves the calling thread without a current context.
fn clear_current_context() {
    // SAFETY: a null context clears this thread's current context. The
    // primary context remains retained by the runtime.
    unsafe { cudarc::driver::result::ctx::set_current(core::ptr::null_mut()).unwrap() }
}

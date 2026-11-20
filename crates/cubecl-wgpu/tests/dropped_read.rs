//! A read whose future is dropped before it resolves leaves its staging buffer mapped,
//! or with a map still pending.
//!
//! The buffer must not go back to the staging memory while still in either of those states
//! otherwise wgpu will reject any subsequent read copying into it on submit.

use core::pin::pin;
use core::task::{Context, Waker};
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[test]
fn a_dropped_read_does_not_poison_the_next_one() {
    let client = <WgpuRuntime>::client(&WgpuDevice::default());
    let input = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let handle = client.create_from_slice(&input);

    for _ in 0..4 {
        let mut read = pin!(client.read_async(vec![handle.clone()]));
        let _ = read.as_mut().poll(&mut Context::from_waker(Waker::noop()));
    }

    let bytes = client.read_one(handle).expect("the read must succeed");
    assert_eq!(&bytes[..], &input);
}

#[cfg(target_has_atomic = "ptr")]
use alloc::{sync::Arc, task::Wake};

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::{task::Wake, Arc};

use core::{
    future::Future,
    task::{Context, Poll, Waker},
};

struct DummyWaker;

#[cfg(target_has_atomic = "ptr")]
impl Wake for DummyWaker {
    fn wake(self: Arc<Self>) {}
    fn wake_by_ref(self: &Arc<Self>) {}
}

#[cfg(not(target_has_atomic = "ptr"))]
impl Wake for DummyWaker {
    fn wake(_this: Arc<Self>) {}
    fn wake_by_ref(_this: &Arc<Self>) {}
}

/// Read a future synchronously.
///
/// On WASM futures cannot block, so this only succeeds if the future returns immediately.
/// If you want to handle this error, please use
/// try_read_sync instead.
pub fn read_sync<F: Future<Output = T>, T>(f: F) -> T {
    try_read_sync(f).expect("Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM. If possible, try using an async variant of this function instead.")
}

/// Read a future synchronously.
///
/// On WASM futures cannot block, so this only succeeds if the future returns immediately.
/// otherwise this returns None.
pub fn try_read_sync<F: Future<Output = T>, T>(f: F) -> Option<T> {
    // Create a dummy context.
    let waker = Waker::from(Arc::new(DummyWaker));
    let mut context = Context::from_waker(&waker);

    // Pin & poll the future. Some backends don't do async readbacks, and instead immediately get
    // the data. This let's us detect when a future is synchronous and doesn't require any waiting.
    let mut pinned = core::pin::pin!(f);

    match pinned.as_mut().poll(&mut context) {
        Poll::Ready(output) => Some(output),
        // On platforms that support it, now just block on the future and drive it to completion.
        #[cfg(all(not(target_family = "wasm"), feature = "std"))]
        Poll::Pending => Some(super::future::block_on(pinned)),
        // Otherwise, just bail and return None - this futures will have to be read back asynchronously.
        #[cfg(any(target_family = "wasm", not(feature = "std")))]
        Poll::Pending => None,
    }
}

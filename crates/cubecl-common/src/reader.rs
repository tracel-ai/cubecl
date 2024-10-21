use core::future::Future;

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
    #[cfg(target_family = "wasm")]
    {
        use core::task::Poll;

        match embassy_futures::poll_once(f) {
            Poll::Ready(output) => Some(output),
            _ => None,
        }
    }
    #[cfg(not(target_family = "wasm"))]
    {
        Some(super::future::block_on(f))
    }
}

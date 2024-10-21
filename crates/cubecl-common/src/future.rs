use alloc::boxed::Box;
use core::future::Future;

/// Block until the [future](Future) is completed and returns the result.
pub fn block_on<O>(fut: impl Future<Output = O>) -> O {
    #[cfg(target_family = "wasm")]
    {
        super::reader::read_sync(fut)
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "std")))]
    {
        embassy_futures::block_on(fut)
    }

    #[cfg(all(not(target_family = "wasm"), feature = "std"))]
    {
        futures_lite::future::block_on(fut)
    }
}

/// Tries to catch panics within the future.
pub async fn catch_unwind<O>(
    future: impl Future<Output = O>,
) -> Result<O, Box<dyn core::any::Any + core::marker::Send>> {
    #[cfg(all(not(target_family = "wasm"), feature = "std"))]
    {
        use core::panic::AssertUnwindSafe;
        use futures_lite::FutureExt;
        AssertUnwindSafe(future).catch_unwind().await
    }

    #[cfg(any(target_family = "wasm", not(feature = "std")))]
    {
        Ok(future.await)
    }
}

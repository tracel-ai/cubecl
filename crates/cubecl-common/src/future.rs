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

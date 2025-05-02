use alloc::boxed::Box;
use core::{future::Future, pin::Pin};

/// A dynamically typed, boxed, future. Useful for futures that need to ensure they
/// are not capturing any of their inputs.
pub type DynFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

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

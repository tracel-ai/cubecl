use alloc::boxed::Box;
use core::{future::Future, pin::Pin};

/// A dynamically typed, boxed, future. Useful for futures that need to ensure they
/// are not capturing any of their inputs.
pub type DynFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Spawns a future on the platform executor.
#[cfg(target_family = "wasm")]
pub fn spawn_detached(fut: impl Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(fut);
}

/// Spawns a future on a background thread.
#[cfg(not(target_family = "wasm"))]
pub fn spawn_detached(fut: impl Future<Output = ()> + Send + 'static) {
    #[cfg(feature = "std")]
    std::thread::spawn(|| block_on(fut));

    #[cfg(not(feature = "std"))]
    {
        drop(fut);
        panic!("spawn_detached requires the `std` feature");
    }
}

/// Block until the [future](Future) is completed and returns the result.
#[cfg_attr(feature = "std", allow(clippy::needless_lifetimes))]
#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(fut)))]
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

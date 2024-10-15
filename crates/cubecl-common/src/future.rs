use alloc::boxed::Box;
use core::{any, future::Future};

/// Block until the [future](Future) is completed and returns the result.
pub fn block_on<O>(fut: impl Future<Output = O>) -> O {
    futures::executor::block_on(fut)
}

/// Tries to catch panics within the future.
pub async fn catch_unwind<O>(
    future: impl Future<Output = O>,
) -> Result<O, Box<dyn any::Any + Send>> {
    #[cfg(not(target_family = "wasm"))]
    {
        use core::panic::AssertUnwindSafe;
        use futures::FutureExt;

        AssertUnwindSafe(future).catch_unwind().await
    }
    #[cfg(target_family = "wasm")]
    Ok(future.await)
}

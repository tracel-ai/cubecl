use alloc::boxed::Box;
use core::{future::Future, pin::Pin};

/// A dynamically typed, boxed, future. Useful for futures that need to ensure they
/// are not capturing any of their inputs.
pub type DynFut<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Spawns a future to run detached. This will use a thread on native, or the browser runtime
/// on WASM. The returned JoinOnDrop will join the thread when it is dropped.
pub fn spawn_detached_fut(fut: impl Future<Output = ()> + Send + 'static) {
    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            wasm_bindgen_futures::spawn_local(fut);
        } else if #[cfg(feature = "std")] {
            std::thread::spawn(|| block_on(fut));
        } else {
            drop(fut); // Just to prevent unused.
            panic!("spawn_detached_fut is only supported with 'std' or on 'wasm' targets");
        }
    }
}

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

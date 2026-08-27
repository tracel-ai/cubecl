pub(crate) mod capture;
pub(crate) mod communication;
pub(crate) mod context;
pub(crate) mod driver;
pub(crate) mod storage;
pub(crate) mod stream;
pub(crate) mod sync;

mod server;

pub use server::*;

/// One unit of work against the device — the shared
/// [`Command`](cubecl_runtime::command::Command), driven by [`driver::Cuda`].
pub(crate) type Command<'a> = cubecl_runtime::command::Command<'a, driver::Cuda>;

/// The graphs this device has captured, driven by [`driver::Cuda`].
pub(crate) type Captures = cubecl_runtime::command::Captures<driver::Cuda>;

/// A capture window on one stream, driven by [`driver::Cuda`].
pub(crate) type Window<'a> = cubecl_runtime::command::Window<'a, driver::Cuda>;

/// Creates a `Vec<I>` of the given length with uninitialized elements.
///
/// # Safety
///
/// The caller must initialize all elements before reading them. Reading uninitialized
/// memory is undefined behavior. `I` must be valid for any bit pattern (e.g. integer types).
#[allow(clippy::uninit_vec)]
pub unsafe fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    // SAFETY: The capacity was set to `len` above, so setting the length to `len` is valid.
    // The caller is responsible for initializing all elements before reading them.
    unsafe {
        data.set_len(len);
    };

    data
}

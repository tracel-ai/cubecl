mod server;

pub(crate) mod command;
pub(crate) mod context;
pub(crate) mod fence;
pub(crate) mod io;
pub(crate) mod storage;
pub(crate) mod stream;

pub use server::*;
pub use storage::*;

pub(crate) const MB: usize = 1024 * 1024;

/// Creates a `Vec<I>` of the given length with uninitialized elements.
///
/// # Safety note
///
/// The caller must initialize all elements before reading them. Reading uninitialized
/// memory is undefined behavior.
#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    // SAFETY: The capacity was set to `len` above, so setting the length to `len` is valid.
    // The caller is responsible for initializing all elements before reading them.
    unsafe {
        data.set_len(len);
    };

    data
}

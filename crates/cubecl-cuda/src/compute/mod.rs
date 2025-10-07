pub(crate) mod command;
pub(crate) mod context;
pub(crate) mod io;
pub(crate) mod storage;
pub(crate) mod stream;
pub(crate) mod sync;

mod server;

pub use server::*;

#[allow(clippy::uninit_vec)]
/// Initialize a vector without any values written in it.
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

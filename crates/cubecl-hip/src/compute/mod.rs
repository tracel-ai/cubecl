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

#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

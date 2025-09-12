pub(crate) mod alloc_controlle;
pub(crate) mod io;
pub(crate) mod storage;
pub(crate) mod sync;

mod data_service;
mod server;

pub use data_service::*;
pub use server::*;

#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

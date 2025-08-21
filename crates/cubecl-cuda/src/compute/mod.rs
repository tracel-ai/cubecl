mod server;
mod storage;
mod data_service;

pub mod sync;

pub use server::*;
pub use storage::*;
pub use data_service::*;

#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

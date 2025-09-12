mod server;

pub(crate) mod alloc_controller;
pub(crate) mod fence;
pub(crate) mod storage;

pub use server::*;
pub use storage::*;

#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

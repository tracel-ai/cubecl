pub(crate) mod fence;
#[cfg(feature = "nccl")]
mod nccl;
mod server;
mod storage;

#[cfg(feature = "nccl")]
pub use nccl::NcclOp;
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

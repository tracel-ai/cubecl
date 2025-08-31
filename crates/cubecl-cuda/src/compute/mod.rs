mod server;
mod storage;


pub(crate) mod fence;

pub use server::*;
pub use storage::*;

#[cfg(test)]
mod testgen;

#[cfg(test)]
pub use testgen::*;

#[allow(clippy::uninit_vec)]
pub fn uninit_vec<I>(len: usize) -> Vec<I> {
    let mut data = Vec::with_capacity(len);

    unsafe {
        data.set_len(len);
    };

    data
}

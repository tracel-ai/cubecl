#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub mod compiler;
pub use device::*;

pub use runtime::HipRuntime;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::HipRuntime;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_all!();
}

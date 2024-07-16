#[macro_use]
extern crate derive_new;

extern crate alloc;

mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime;

    cubecl_core::testgen_all!();
    cubecl_lac::testgen_all!();
}

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod compiler;
mod compute;
mod device;
mod element;
mod graphics;
mod runtime;

pub use compiler::wgsl::WgslCompiler;
pub use compute::*;
pub use device::*;
pub use element::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use compiler::spirv;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime<crate::WgslCompiler>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_all!();
}

#[cfg(all(test, feature = "spirv"))]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime<crate::spirv::VkSpirvCompiler>;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_all!();
}

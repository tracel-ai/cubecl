pub mod branch;
pub mod cmp;
pub mod constant;
pub mod entrypoint;
pub mod func;
pub mod general;
pub mod math;
pub mod memory;
pub mod metadata;
pub mod to_llvm;

mod prelude {
    pub use super::to_llvm::ToLLVMDialect;
    pub use cubecl_core::ir::prelude::*;
}

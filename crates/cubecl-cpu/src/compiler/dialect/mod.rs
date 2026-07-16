pub mod branch;
pub mod constant;
pub mod entrypoint;
pub mod func;
pub mod math;
pub mod to_llvm;

mod prelude {
    pub use super::to_llvm::ToLLVMDialect;
    pub use cubecl_core::ir::prelude::*;
}

mod cyclic_loading;
// mod double_buffered;
mod loader;
mod single_buffered;
mod tilewise_loading;

pub use cyclic_loading::*;
pub use loader::*;
pub use tilewise_loading::*;

// Use the one to be exported as full_load::Matmul
pub use single_buffered::Matmul;
// pub use double_buffered::Matmul;

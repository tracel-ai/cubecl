#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub mod compiler;
pub use device::*;

pub use runtime::*;

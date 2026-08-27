//! One unit of work against the device, and what a backend supplies for it.

mod backend;
mod base;
mod capture;
mod staging;

pub use backend::*;
pub use base::*;
pub use capture::*;
pub use staging::*;

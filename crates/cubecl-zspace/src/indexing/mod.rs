//! Indexing Utilities

mod type_conversion;
mod wrapping;

pub use type_conversion::{AsIndex, AsSize};
pub use wrapping::{IndexWrap, ravel_index, wrap_index};

mod contiguous;
mod contiguous_lined;

mod handle;
pub mod identity;
mod matrix_batch_layout;

pub use contiguous::*;
pub use contiguous_lined::*;
pub use handle::*;
pub use identity::*;
pub use matrix_batch_layout::*;
pub use view::*;

pub mod layout;
pub mod view;
pub mod r#virtual;

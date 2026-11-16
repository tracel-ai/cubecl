//! An operand's dimensions mapped onto its buffer's physical axes: the [`Projection`] stated on
//! the host, one [`PhysicalAxisMap`] per buffer axis, and its evaluation in the kernel.

pub(crate) mod axes_builder;
pub(crate) mod axis_map;
pub(crate) mod base;
pub(crate) mod buffer_layout;
pub(crate) mod carriers;
pub(crate) mod in_kernel;
pub(crate) mod runtime_map;
pub(crate) mod witness;

pub(crate) use axis_map::*;
pub(crate) use buffer_layout::*;
pub(crate) use carriers::*;
pub(crate) use in_kernel::*;
pub(crate) use runtime_map::*;
pub(crate) use witness::Witness;

pub use axes_builder::{AxesBuilder, split};
pub use axis_map::{Divisor, Offset, PhysicalAxisMap, Scale};
pub use base::Projection;

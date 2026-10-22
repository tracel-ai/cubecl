//! An operand's logical axes mapped onto its buffer's physical axes: the [`Projection`] stated on
//! the host, one [`PhysicalAxisMap`] per buffer dim, and its evaluation in the kernel.

pub(crate) mod axis_map;
pub(crate) mod base;
pub(crate) mod buffer_layout;
pub(crate) mod dims_builder;
pub(crate) mod in_kernel;
pub(crate) mod runtime_map;
pub(crate) mod witness;

pub(crate) use axis_map::*;
pub(crate) use buffer_layout::*;
pub(crate) use in_kernel::*;
pub(crate) use runtime_map::*;
pub(crate) use witness::Witness;

pub use axis_map::{Divisor, Offset, PhysicalAxisMap, Scale};
pub use base::Projection;
pub use dims_builder::{DimsBuilder, split};

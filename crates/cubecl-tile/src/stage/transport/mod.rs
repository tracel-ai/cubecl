//! Transports that move an operand's cells into the tile an instruction reads them from.

pub(crate) mod async_copy;
pub(crate) mod base;
pub(crate) mod padded;
pub(crate) mod prefetch;
pub(crate) mod scaled;
pub(crate) mod scanned;
pub(crate) mod straight;

pub(crate) use padded::Padding;
pub(crate) use prefetch::{MOST_FETCHED_SCALARS, UnitLines};

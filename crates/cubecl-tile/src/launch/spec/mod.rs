//! The comptime half of an operand ([`TileSpec`]) and the policies it carries: how its reads are
//! bounds-checked ([`BoundaryPolicy`]) and who moves it into a stage ([`Delivery`]).

pub(crate) mod base;
pub(crate) mod boundary_policy;
pub(crate) mod delivery;

pub use base::TileSpec;
pub use boundary_policy::BoundaryPolicy;
pub use delivery::Delivery;

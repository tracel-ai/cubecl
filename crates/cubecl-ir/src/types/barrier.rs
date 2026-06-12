use cubecl_macros_internal::TypeHash;
use pliron::derive::{format, pliron_type};

use crate::interfaces::{aligned, sized};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
#[format]
pub enum BarrierLevel {
    #[format("`unit`")]
    Unit,
    #[format("`cube`")]
    Cube,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.barrier",
    format = "`barrier`",
    generate_get = true,
    verifier = "succ"
)]
pub struct BarrierType;
aligned!(BarrierType, align_of::<u64>());
sized!(BarrierType, size_of::<u64>());

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.barrier_token",
    format = "`barrier_token`",
    generate_get = true,
    verifier = "succ"
)]
pub struct BarrierTokenType;
aligned!(BarrierTokenType, align_of::<u64>());
sized!(BarrierTokenType, size_of::<u64>());

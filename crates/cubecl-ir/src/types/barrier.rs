use cubecl_macros_internal::TypeHash;
use derive_more::Deref;
use pliron::{
    derive::{format, pliron_type},
    r#type::TypedHandle,
};

use crate::{aligned, sized};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
#[format]
pub enum BarrierLevel {
    Unit,
    Cube,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.barrier",
    format = "`barrier<` $0 `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct BarrierType(pub BarrierLevel);
aligned!(BarrierType, align_of::<u64>());
sized!(BarrierType, size_of::<u64>());

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deref)]
#[pliron_type(
    name = "cube.barrier_token",
    format = "`barrier_token<` $0 `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct BarrierTokenType(pub TypedHandle<BarrierType>);
aligned!(BarrierTokenType, align_of::<u64>());
sized!(BarrierTokenType, size_of::<u64>());

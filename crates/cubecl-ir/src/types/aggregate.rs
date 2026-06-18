use cubecl_macros_internal::TypeHash;
use derive_more::Display;
use pliron::derive::{format, pliron_type, type_interface_impl};

use crate::{interfaces::AggregateType, prelude::*, types::scalar::IndexType};

#[pliron_type(
    name = "cube.ptr_aggregate",
    format = "$meta `<` $base_ty `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct PtrAggregateType {
    pub base_ty: TypeHandle,
    pub meta: MetadataKind,
}

#[type_interface_impl]
impl AggregateType for PtrAggregateType {
    fn field_ty(&self, ctx: &Context, field_idx: usize) -> TypeHandle {
        match self.meta {
            MetadataKind::Slice => match field_idx {
                0 => self.base_ty,
                1 | 2 => IndexType::get(ctx).into(),
                _ => panic!("Invalid index"),
            },
            MetadataKind::BoundsCheck => match field_idx {
                0 => self.base_ty,
                1 => IndexType::get(ctx).into(),
                _ => panic!("Invalid index"),
            },
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord, Display)]
#[format]
pub enum MetadataKind {
    /// Slice metadata (offset and length)
    #[display("slice")]
    Slice,
    /// Bounds check (in bounds)
    #[display("bounds_checked")]
    BoundsCheck,
}

pub struct BoundsCheckMetadata;
impl BoundsCheckMetadata {
    pub const POINTER: usize = 0;
    pub const IS_IN_BOUNDS: usize = 1;
}

pub struct SliceMetadata;
impl SliceMetadata {
    pub const LIST: usize = 0;
    pub const OFFSET: usize = 1;
    pub const LENGTH: usize = 2;
}

use alloc::boxed::Box;
use pliron::{
    attribute::AttrObj,
    derive::{pliron_type, type_interface_impl},
    utils::table::HMap,
};

use crate::{
    attributes::IndexAttr,
    interfaces::memory_slot::DestructurableTypeInterface,
    prelude::*,
    types::scalar::{BoolType, IndexType},
};

#[pliron_type(
    name = "cube.slice",
    format = "`<` $base_ty `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct SliceType {
    pub base_ty: TypeHandle,
}

#[type_interface_impl]
impl DestructurableTypeInterface for SliceType {
    fn subelement_index_map(&self, ctx: &Context) -> Option<HMap<AttrObj, TypeHandle>> {
        let mut out = HMap::new();
        out.insert(index_attr(0), self.base_ty);
        out.insert(index_attr(1), IndexType::get(ctx).to_handle());
        out.insert(index_attr(2), IndexType::get(ctx).to_handle());
        Some(out)
    }

    fn type_at_index(&self, ctx: &Context, index: &AttrObj) -> TypeHandle {
        match index.downcast_ref::<IndexAttr>().unwrap().0 {
            0 => self.base_ty,
            1 | 2 => IndexType::get(ctx).to_handle(),
            _ => unreachable!(),
        }
    }
}

#[pliron_type(
    name = "cube.checked_ptr",
    format = "`<` $base_ty `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CheckedPtrType {
    pub base_ty: TypeHandle,
}

#[type_interface_impl]
impl DestructurableTypeInterface for CheckedPtrType {
    fn subelement_index_map(&self, ctx: &Context) -> Option<HMap<AttrObj, TypeHandle>> {
        let mut out = HMap::new();
        out.insert(index_attr(0), self.base_ty);
        out.insert(index_attr(1), BoolType::get(ctx).to_handle());
        Some(out)
    }

    fn type_at_index(&self, ctx: &Context, index: &AttrObj) -> TypeHandle {
        match index.downcast_ref::<IndexAttr>().unwrap().0 {
            0 => self.base_ty,
            1 => BoolType::get(ctx).to_handle(),
            _ => unreachable!(),
        }
    }
}

pub fn index_attr(idx: usize) -> AttrObj {
    Box::new(IndexAttr(idx))
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

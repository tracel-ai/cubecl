use derive_more::From;
use derive_new::new;
use pliron::{
    builtin::attr_interfaces::TypedAttrInterface,
    context::{Context, Ptr},
    derive::{attr_interface_impl, pliron_attr},
    r#type::{TypeObj, TypePtr},
    utils::apfloat::Double,
};

use crate::types::scalar::{BoolType, FloatType, IndexType, IntType, UIntType};

#[pliron_attr(name = "cube.index", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Debug, Hash, PartialOrd, Ord)]
pub struct IndexAttr(pub usize);

impl From<IndexAttr> for usize {
    fn from(value: IndexAttr) -> Self {
        value.0
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for IndexAttr {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        IndexType::get(ctx).into()
    }
}

/// A boolean attribute
#[pliron_attr(name = "cube.bool", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct BoolAttr(pub bool);

impl BoolAttr {
    /// Create a new [`BoolAttr`].
    pub fn new(value: bool) -> Self {
        BoolAttr(value)
    }
}

impl From<BoolAttr> for bool {
    fn from(value: BoolAttr) -> Self {
        value.0
    }
}

impl From<bool> for BoolAttr {
    fn from(value: bool) -> Self {
        BoolAttr::new(value)
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for BoolAttr {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        BoolType::get(ctx).into()
    }
}

#[pliron_attr(name = "cube.int", format = "$val `: ` $ty", verifier = "succ")]
#[derive(new, PartialEq, Eq, Clone, Debug, Hash)]
pub struct IntAttr {
    pub ty: TypePtr<IntType>,
    pub val: i64,
}

#[attr_interface_impl]
impl TypedAttrInterface for IntAttr {
    fn get_type(&self, _ctx: &Context) -> Ptr<TypeObj> {
        self.ty.into()
    }
}

#[pliron_attr(name = "cube.uint", format = "$val `: ` $ty", verifier = "succ")]
#[derive(new, PartialEq, Eq, Clone, Debug, Hash)]
pub struct UIntAttr {
    pub ty: TypePtr<UIntType>,
    pub val: u64,
}

#[attr_interface_impl]
impl TypedAttrInterface for UIntAttr {
    fn get_type(&self, _ctx: &Context) -> Ptr<TypeObj> {
        self.ty.into()
    }
}

#[pliron_attr(name = "cube.float", format = "$val `: ` $ty", verifier = "succ")]
#[derive(new, PartialEq, Clone, Debug)]
pub struct FloatAttr {
    pub ty: TypePtr<FloatType>,
    pub val: Double,
}

#[attr_interface_impl]
impl TypedAttrInterface for FloatAttr {
    fn get_type(&self, _ctx: &Context) -> Ptr<TypeObj> {
        self.ty.into()
    }
}

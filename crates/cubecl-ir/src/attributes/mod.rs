use alloc::boxed::Box;

use derive_more::From;
use derive_new::new;
use num_traits::NumCast;
use pliron::{
    builtin::{
        attr_interfaces::{MaterializableAttr, TypedAttrInterface},
        ops::ConstantOp,
    },
    context::{Context, Ptr},
    derive::{attr_interface_impl, pliron_attr},
    op::Op,
    operation::Operation,
    r#type::{TypeObj, TypePtr},
    utils::apfloat::{Double, double_to_f64, f64_to_double},
};

use crate::{
    FloatKind,
    types::scalar::{BoolType, FloatType, IndexType, IntType, UIntType},
};

macro_rules! materialize_const {
    ($ty: ty) => {
        #[attr_interface_impl]
        impl MaterializableAttr for $ty {
            fn materialize(&self, ctx: &mut Context) -> Ptr<Operation> {
                let const_op = ConstantOp::new(ctx, Box::new(self.clone()));
                const_op.get_operation()
            }
        }
    };
}

#[pliron_attr(name = "cube.index", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord)]
pub struct IndexAttr(pub usize);
materialize_const!(IndexAttr);

impl IndexAttr {
    pub fn value(&self) -> Option<usize> {
        Some(self.0)
    }

    pub fn with_value(&self, new_val: usize) -> Self {
        Self::new(new_val)
    }
}

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
#[derive(new, PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct BoolAttr(pub bool);
materialize_const!(BoolAttr);

impl BoolAttr {
    pub fn value(&self) -> Option<bool> {
        Some(self.0)
    }

    pub fn with_value(&self, new_val: bool) -> Self {
        Self::new(new_val)
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
materialize_const!(IntAttr);

impl IntAttr {
    pub fn value<T: NumCast + TypedLiteral>(&self, ctx: &Context) -> Option<T> {
        if T::is_same_type(ctx, self.ty.into()) {
            Some(T::from(self.val).expect("Should succeed"))
        } else {
            None
        }
    }

    pub fn with_value<T: NumCast>(&self, new_val: T) -> Self {
        Self::new(self.ty, new_val.to_i64().unwrap())
    }
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
materialize_const!(UIntAttr);

impl UIntAttr {
    pub fn value<T: NumCast + TypedLiteral>(&self, ctx: &Context) -> Option<T> {
        if T::is_same_type(ctx, self.ty.into()) {
            Some(T::from(self.val).expect("Should succeed"))
        } else {
            None
        }
    }

    pub fn with_value<T: NumCast>(&self, new_val: T) -> Self {
        Self::new(self.ty, new_val.to_u64().expect("Should convert"))
    }
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
materialize_const!(FloatAttr);

impl FloatAttr {
    pub fn value<T: NumCast + TypedLiteral>(&self, ctx: &Context) -> Option<T> {
        if T::is_same_type(ctx, self.ty.into()) {
            Some(T::from(double_to_f64(self.val)).expect("Should succeed"))
        } else {
            None
        }
    }

    pub fn with_value<T: NumCast>(&self, new_val: T) -> Self {
        Self::new(
            self.ty,
            f64_to_double(new_val.to_f64().expect("Should convert")),
        )
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for FloatAttr {
    fn get_type(&self, _ctx: &Context) -> Ptr<TypeObj> {
        self.ty.into()
    }
}

pub trait TypedLiteral {
    fn is_same_type(ctx: &Context, ty: Ptr<TypeObj>) -> bool;
}

macro_rules! literal {
    ($ty: ty, $ir_ty: ty, $pred: expr) => {
        impl TypedLiteral for $ty {
            fn is_same_type(ctx: &Context, ty: Ptr<TypeObj>) -> bool {
                ty.deref(ctx).downcast_ref::<$ir_ty>().is_some_and($pred)
            }
        }
    };
    ($ty: ty, $ir_ty: ty) => {
        literal!($ty, $ir_ty, |_| true);
    };
}

literal!(usize, IndexType);

literal!(i8, IntType, |it| it.width == 8);
literal!(i16, IntType, |it| it.width == 16);
literal!(i32, IntType, |it| it.width == 32);
literal!(i64, IntType, |it| it.width == 64);

literal!(u8, UIntType, |it| it.width == 8);
literal!(u16, UIntType, |it| it.width == 16);
literal!(u32, UIntType, |it| it.width == 32);
literal!(u64, UIntType, |it| it.width == 64);

literal!(half::f16, FloatType, |it| it.encoding == FloatKind::F16);
literal!(half::bf16, FloatType, |it| it.encoding == FloatKind::BF16);
literal!(f32, FloatType, |it| it.encoding == FloatKind::F32);
literal!(f64, FloatType, |it| it.encoding == FloatKind::F64);

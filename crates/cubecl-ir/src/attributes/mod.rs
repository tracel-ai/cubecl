use alloc::boxed::Box;

use derive_more::From;
use derive_new::new;
use num_traits::{AsPrimitive, NumCast};
use pliron::{
    builtin::{
        attr_interfaces::{MaterializableAttr, TypedAttrInterface},
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::IntegerType,
    },
    context::{Context, Ptr},
    derive::{attr_interface_impl, pliron_attr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::{
        apfloat::{Double, double_to_f64, f64_to_double},
        apint::{APInt, bw},
    },
};

use crate::{ConstantValue, interfaces::ConstantAttr, settings::Dim3, types::scalar::*};

mod entrypoint;

pub use entrypoint::*;

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

#[macro_export]
macro_rules! ext_attribute {
    ($name: ident: $ty: ty, $($implementors: ty),*) => {
        paste::paste! {
            dict_key!([<ATTR_KEY_ $name:upper>], stringify!($name));

            #[op_interface]
            pub trait [<$name:upper:camel> Interface] {
                fn [<get_ $name>]<'a>(&self, ctx: &'a pliron::context::Context) -> Option<core::cell::Ref<'a, $ty>> {
                    let self_op = self.get_operation().deref(ctx);
                    Ref::filter_map(self_op, |self_op| {
                        self_op
                        .attributes
                        .get::<$ty>(&[<ATTR_KEY_ $name:upper>])
                    }).ok()
                }

                fn [<set_ $name>](&self, ctx: &mut Context, value: $ty) {
                    let mut self_op = self.get_operation().deref_mut(ctx);
                    self_op.attributes.set([<ATTR_KEY_ $name:upper>].clone(), value);
                }

                fn verify(_op: &dyn pliron::op::Op, _ctx: &pliron::context::Context) -> pliron::result::Result<()>
                where
                    Self: Sized,
                {
                    Ok(())
                }
            }
        }
    };
}

#[pliron_attr(name = "cube.index", format = "$0", verifier = "succ")]
#[derive(new, From, PartialEq, Eq, Clone, Copy, Debug, Hash, PartialOrd, Ord)]
pub struct IndexAttr(pub usize);
materialize_const!(IndexAttr);

impl IndexAttr {
    pub fn as_value(&self, _ctx: &Context) -> Option<usize> {
        Some(self.0)
    }

    pub fn with_value(&self, _ctx: &Context, new_val: usize) -> Self {
        Self::new(new_val)
    }
}

#[attr_interface_impl]
impl ConstantAttr for IndexAttr {
    fn as_const_val(&self, _ctx: &Context) -> ConstantValue {
        ConstantValue::UInt(self.0 as u64)
    }
}

impl From<IndexAttr> for usize {
    fn from(value: IndexAttr) -> Self {
        value.0
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for IndexAttr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        IndexType::get(ctx).into()
    }
}

/// A boolean attribute
#[pliron_attr(name = "cube.bool", format = "$0", verifier = "succ")]
#[derive(new, PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct BoolAttr(pub bool);
materialize_const!(BoolAttr);

impl BoolAttr {
    pub fn as_value(&self, _ctx: &Context) -> Option<bool> {
        Some(self.0)
    }

    pub fn with_value(&self, _ctx: &Context, new_val: bool) -> Self {
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
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        BoolType::get(ctx).into()
    }
}

#[attr_interface_impl]
impl ConstantAttr for BoolAttr {
    fn as_const_val(&self, _ctx: &Context) -> ConstantValue {
        ConstantValue::Bool(self.0)
    }
}

pub trait IntAttrExt {
    fn as_value<T>(&self, ctx: &Context) -> Option<T>
    where
        T: TypedLiteral + Copy + 'static,
        i128: AsPrimitive<T>;

    fn with_value<T: NumCast>(&self, ctx: &Context, new_val: T) -> Self;
}

impl IntAttrExt for IntegerAttr {
    fn as_value<T>(&self, ctx: &Context) -> Option<T>
    where
        T: TypedLiteral + Copy + 'static,
        i128: AsPrimitive<T>,
    {
        if T::is_same_type(ctx, self.get_type().into()) {
            Some(self.value().to_i128().as_())
        } else {
            None
        }
    }

    fn with_value<T: NumCast>(&self, ctx: &Context, new_val: T) -> Self {
        let width = bw(self.get_type().deref(ctx).width() as usize);
        let val = new_val.to_i128().expect("Should succeed");
        Self::new(self.get_type(), APInt::from_i128(val, width))
    }
}

#[attr_interface_impl]
impl ConstantAttr for IntegerAttr {
    fn as_const_val(&self, ctx: &Context) -> ConstantValue {
        if self.get_type().deref(ctx).is_signed() {
            ConstantValue::Int(self.value().to_i64())
        } else {
            ConstantValue::UInt(self.value().to_u64())
        }
    }
}

#[pliron_attr(name = "cube.float", format = "$val `: ` $ty", verifier = "succ")]
#[derive(new, PartialEq, Clone, Debug)]
pub struct FloatAttr {
    pub ty: TypeHandle,
    pub val: Double,
}
materialize_const!(FloatAttr);

impl FloatAttr {
    pub fn as_value<T: NumCast + TypedLiteral>(&self, ctx: &Context) -> Option<T> {
        if T::is_same_type(ctx, self.ty) {
            Some(T::from(double_to_f64(self.val)).expect("Should succeed"))
        } else {
            None
        }
    }

    pub fn with_value<T: NumCast>(&self, _ctx: &Context, new_val: T) -> Self {
        Self::new(
            self.ty,
            f64_to_double(new_val.to_f64().expect("Should convert")),
        )
    }
}

#[pliron_attr(name = "cube.dim3", format, verifier = "succ")]
#[derive(new, From, PartialEq, Clone, Debug)]
pub struct Dim3Attr(pub Dim3);

#[attr_interface_impl]
impl TypedAttrInterface for FloatAttr {
    fn get_type(&self, _ctx: &Context) -> TypeHandle {
        self.ty
    }
}

#[attr_interface_impl]
impl ConstantAttr for FloatAttr {
    fn as_const_val(&self, _ctx: &Context) -> ConstantValue {
        ConstantValue::Float(double_to_f64(self.val))
    }
}

pub trait TypedLiteral {
    fn is_same_type(ctx: &Context, ty: TypeHandle) -> bool;
}

macro_rules! literal {
    ($ty: ty, $ir_ty: ty, $pred: expr) => {
        impl TypedLiteral for $ty {
            fn is_same_type(ctx: &Context, ty: TypeHandle) -> bool {
                ty.deref(ctx).downcast_ref::<$ir_ty>().is_some_and($pred)
            }
        }
    };
    ($ty: ty, $ir_ty: ty) => {
        literal!($ty, $ir_ty, |_| true);
    };
}

literal!(usize, IndexType);

literal!(i8, IntegerType, |it| it.width() == 8);
literal!(i16, IntegerType, |it| it.width() == 16);
literal!(i32, IntegerType, |it| it.width() == 32);
literal!(i64, IntegerType, |it| it.width() == 64);

literal!(u8, IntegerType, |it| it.width() == 8);
literal!(u16, IntegerType, |it| it.width() == 16);
literal!(u32, IntegerType, |it| it.width() == 32);
literal!(u64, IntegerType, |it| it.width() == 64);

literal!(half::f16, Float16Type);
literal!(half::bf16, BFloat16Type);
literal!(f32, Float32Type);
literal!(f64, Float64Type);

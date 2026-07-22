use crate::{
    ConstantValue, ElemType,
    dialect::synchronization::SyncScope,
    prelude::*,
    types::{AtomicType, PointerType, VectorType, scalar::*},
};
use pliron::{
    alloc::vec::Vec,
    attribute::{AttrObj, AttributeDict},
    builtin::{attr_interfaces::TypedAttrInterface, ops::ConstantOp, types::IntegerType},
    context::Context,
    derive::{op_interface, type_interface},
    opts::dce::SideEffects,
    r#type::{TypeHandle, type_cast},
    value::Use,
};

pub mod aliasing;

#[macro_export]
macro_rules! verify_op_succ {
    () => {
        fn verify(
            _op: &dyn pliron::op::Op,
            _ctx: &pliron::context::Context,
        ) -> pliron::result::Result<()>
        where
            Self: Sized,
        {
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! verify_ty_succ {
    () => {
        fn verify(
            _op: &dyn pliron::r#type::Type,
            _ctx: &pliron::context::Context,
        ) -> pliron::result::Result<()>
        where
            Self: Sized,
        {
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! verify_attr_succ {
    () => {
        fn verify(
            _op: &dyn pliron::attribute::Attribute,
            _ctx: &pliron::context::Context,
        ) -> pliron::result::Result<()>
        where
            Self: Sized,
        {
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! Pure {
    ($ty: ty) => {
        $crate::NoSideEffects!($ty);
        $crate::NoMemoryEffect!($ty);
    };
}

#[op_interface]
pub trait TriviallyUnrollable: MaterializableOp {
    verify_op_succ!();
}

/// Op that can be rematerialized from a set of operands.
/// Should be implemented for anything that doesn't have regions or successors.
#[op_interface]
pub trait MaterializableOp {
    verify_op_succ!();
    fn materialize(
        &self,
        ctx: &mut Context,
        result_ty: Vec<TypeHandle>,
        operands: Vec<Value>,
        attributes: AttributeDict,
    ) -> Ptr<Operation>;
}

#[macro_export]
macro_rules! CanMaterialize {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::MaterializableOp for $ty {
            fn materialize(
                &self,
                ctx: &mut Context,
                result_ty: Vec<TypeHandle>,
                operands: Vec<Value>,
                attributes: AttributeDict,
            ) -> Ptr<Operation> {
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    result_ty,
                    operands,
                    vec![],
                    0,
                );
                op.deref_mut(ctx).attributes = attributes;
                op
            }
        }
    };
}

CanMaterialize!(ConstantOp);

#[op_interface]
pub trait Synchronizes: SideEffects {
    verify_op_succ!();

    fn scope(&self, ctx: &Context) -> SyncScope;
}

macro_rules! synchronizes {
    ($ty: ty, $scope: expr) => {
        #[::pliron::derive::op_interface_impl]
        impl crate::interfaces::Synchronizes for $ty {
            #[allow(unused_variables)]
            fn scope(&self, ctx: &::pliron::context::Context) -> SyncScope {
                $scope
            }
        }
        #[pliron::derive::op_interface_impl]
        impl pliron::opts::dce::SideEffects for $ty {
            fn has_side_effects(&self, _ctx: &Context) -> bool {
                true
            }
        }
    };
}
pub(crate) use synchronizes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEffect {
    Read(Value),
    Write(Value),
    ReadAll,
    WriteAll,
}

#[op_interface]
pub trait MemoryEffects {
    verify_op_succ!();
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect>;
}

#[macro_export]
macro_rules! NoMemoryEffect {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::MemoryEffects for $ty {
            fn memory_effects(
                &self,
                _ctx: &pliron::context::Context,
            ) -> $crate::alloc::vec::Vec<$crate::interfaces::MemoryEffect> {
                $crate::alloc::vec![]
            }
        }
    };
}

NoMemoryEffect!(ConstantOp);

#[type_interface]
pub trait AlignedType {
    verify_ty_succ!();

    fn align(&self, ctx: &Context) -> usize;
}

#[macro_export]
macro_rules! aligned {
    ($ty: ty, $align: expr) => {
        #[::pliron::derive::type_interface_impl]
        impl $crate::interfaces::AlignedType for $ty {
            #[allow(unused_variables)]
            fn align(&self, ctx: &::pliron::context::Context) -> usize {
                $align
            }
        }
    };
}

#[type_interface]
pub trait SizedType: AlignedType {
    verify_ty_succ!();
    fn size(&self, ctx: &Context) -> usize;
    fn size_bits(&self, ctx: &Context) -> usize {
        self.size(ctx) * 8
    }
}

#[macro_export]
macro_rules! sized {
    ($ty: ty, $size: expr) => {
        #[::pliron::derive::type_interface_impl]
        impl $crate::interfaces::SizedType for $ty {
            #[allow(unused_variables)]
            fn size(&self, ctx: &::pliron::context::Context) -> usize {
                $size
            }
        }
    };
}

#[macro_export]
macro_rules! NoSideEffects {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl pliron::opts::dce::SideEffects for $ty {
            fn has_side_effects(&self, _ctx: &Context) -> bool {
                false
            }
        }
    };
}

#[type_interface]
pub trait MaybeVectorizedType {
    verify_ty_succ!();

    fn vector_size(&self, ctx: &Context) -> usize;
    fn try_vector_size(&self, ctx: &Context) -> Option<usize> {
        Some(self.vector_size(ctx))
    }
}

#[macro_export]
macro_rules! scalar {
    ($ty: ty) => {
        #[::pliron::derive::type_interface_impl]
        impl $crate::interfaces::MaybeVectorizedType for $ty {
            fn vector_size(&self, _ctx: &::pliron::context::Context) -> usize {
                1
            }
        }

        #[::pliron::derive::type_interface_impl]
        impl $crate::interfaces::ScalarizableType for $ty {
            fn scalar_type(&self, ctx: &Context) -> ::pliron::r#type::TypeHandle {
                use ::pliron::r#type::Type;
                self.get_self_handle(ctx)
            }
        }

        #[::pliron::derive::type_interface_impl]
        impl $crate::interfaces::HasElementType for $ty {
            fn element_type(&self, ctx: &Context) -> Option<::pliron::r#type::TypeHandle> {
                use ::pliron::r#type::Type;
                Some(self.get_self_handle(ctx))
            }
        }
    };
}

#[type_interface]
pub trait MaybePackedType {
    verify_ty_succ!();

    fn packing_factor(&self, ctx: &Context) -> usize;
}

macro_rules! not_packed {
    ($ty: ty) => {
        #[::pliron::derive::type_interface_impl]
        impl crate::interfaces::MaybePackedType for $ty {
            fn packing_factor(&self, _ctx: &::pliron::context::Context) -> usize {
                1
            }
        }
    };
}
pub(crate) use not_packed;

#[type_interface]
pub trait ScalarizableType {
    verify_ty_succ!();
    fn scalar_type(&self, ctx: &Context) -> TypeHandle;
}

#[type_interface]
pub trait ScalarType {
    verify_ty_succ!();
    fn elem_type(&self, ctx: &Context) -> ElemType;
}

#[type_interface]
pub trait AggregateType {
    verify_ty_succ!();

    fn field_ty(&self, ctx: &Context, field_idx: usize) -> TypeHandle;
}

#[type_interface]
pub trait IndexableType {
    verify_ty_succ!();

    fn indexed_type(&self, ctx: &Context) -> TypeHandle;
}

#[type_interface]
pub trait HasElementType {
    verify_ty_succ!();
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle>;
}

#[op_interface]
pub trait SimplifyInterface {
    verify_op_succ!();
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Option<Value>;
}

#[attr_interface]
pub trait ConstantAttr: TypedAttrInterface {
    verify_attr_succ!();
    fn as_const_val(&self, ctx: &Context) -> ConstantValue;
}

#[macro_export]
macro_rules! try_cast_ty {
    ($ty: expr, $ctx: expr, $interface: ty) => {
        type_cast::<$interface>(&*$ty)
            .ok_or_else(|| {
                $crate::alloc::format!(
                    "Expected type {} {} to implement {}",
                    $ty.get_type_id(),
                    $ty.disp($ctx),
                    stringify!($interface)
                )
            })
            .unwrap()
    };
}

#[macro_export]
macro_rules! try_cast_op {
    ($op: expr, $ctx: expr, $interface: ty) => {
        op_cast::<$interface>(&*$op)
            .ok_or_else(|| {
                $crate::alloc::format!(
                    "Expected op {} {} to implement {}",
                    $op.get_opid(),
                    $op.disp($ctx),
                    stringify!($interface)
                )
            })
            .unwrap()
    };
}

#[macro_export]
macro_rules! match_ty {
    (($handle: expr) { $($ty: ty => $body: expr,)*; _ => $default: expr }) => {
        (|| {
            $(if $handle.is::<$ty>() {
                return $body;
            })*
            $default
        })()
    };
    (($handle: expr) { $($ty: ty => $body: expr,)* }) => {
        (|| {
            $(if $handle.is::<$ty>() {
                return $body;
            })*
            unreachable!()
        })()
    };
}

pub trait TypedExt: Typed {
    fn size(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let sized = try_cast_ty!(ty, ctx, dyn SizedType);
        sized.size(ctx)
    }

    fn size_bits(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let sized = try_cast_ty!(ty, ctx, dyn SizedType);
        sized.size_bits(ctx)
    }

    fn unpacked_size_bits(&self, ctx: &Context) -> usize {
        self.element_ty(ctx).scalar_ty(ctx).size_bits(ctx) / self.packing_factor(ctx)
    }

    fn align(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let aligned = try_cast_ty!(ty, ctx, dyn AlignedType);
        aligned.align(ctx)
    }

    fn is_ptr(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<PointerType>()
    }

    fn is_atomic(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<AtomicType>()
    }

    fn is_vector(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<VectorType>()
    }

    fn is_vector_of_size(&self, ctx: &Context, size: usize) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<VectorType>()
            .is_some_and(|it| it.vectorization == size)
    }

    fn is_immutable(&self, ctx: &Context) -> bool {
        !self.is_ptr(ctx)
    }

    fn vector_size(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = try_cast_ty!(ty, ctx, dyn MaybeVectorizedType);
        maybe_vec.vector_size(ctx)
    }

    fn try_get_vector_size(&self, ctx: &Context) -> Option<usize> {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = type_cast::<dyn MaybeVectorizedType>(&*ty)?;
        maybe_vec.try_vector_size(ctx)
    }

    fn packing_factor(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_packed = try_cast_ty!(ty, ctx, dyn MaybePackedType);
        maybe_packed.packing_factor(ctx)
    }

    fn scalar_ty(&self, ctx: &Context) -> TypeHandle {
        let ty = self.element_ty(ctx).deref(ctx);
        let scalarizable = try_cast_ty!(ty, ctx, dyn ScalarizableType);
        scalarizable.scalar_type(ctx)
    }

    fn element_ty(&self, ctx: &Context) -> TypeHandle {
        let ty = self.get_type(ctx).deref(ctx);
        let has_element_type = try_cast_ty!(ty, ctx, dyn HasElementType);
        has_element_type
            .element_type(ctx)
            .expect("Expected element type to be some")
    }

    fn unwrap_ptr(&self, ctx: &Context) -> TypeHandle {
        if let Some(ptr) = self.get_type(ctx).deref(ctx).downcast_ref::<PointerType>() {
            ptr.inner
        } else {
            self.get_type(ctx)
        }
    }

    fn try_get_scalar_ty(&self, ctx: &Context) -> Option<TypeHandle> {
        let ty = self.get_type(ctx).deref(ctx);
        let scalarizable = type_cast::<dyn ScalarizableType>(&*ty)?;
        Some(scalarizable.scalar_type(ctx))
    }

    fn try_get_scalar_elem_ty(&self, ctx: &Context) -> Option<TypeHandle> {
        let ty = self.get_type(ctx).deref(ctx);
        let has_elem = type_cast::<dyn HasElementType>(&*ty)?;
        let ty = has_elem.element_type(ctx)?.deref(ctx);
        let scalarizable = type_cast::<dyn ScalarizableType>(&*ty)?;
        Some(scalarizable.scalar_type(ctx))
    }

    fn is_index(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<IndexType>()
    }

    fn is_int(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.is::<IntegerType>()
    }

    fn is_signed_int(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<IntegerType>()
            .is_some_and(|it| it.is_signed())
    }

    fn is_unsigned_int(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<IntegerType>()
            .is_some_and(|it| !it.is_signed())
    }

    fn is_int_of_width(&self, ctx: &Context, width: usize) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<IntegerType>()
            .is_some_and(|it| it.width() as usize == width)
    }

    fn is_float64(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<Float64Type>()
    }

    fn is_float32(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<Float32Type>()
    }

    fn is_tfloat32(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<TFloat32Type>()
    }

    fn is_float16(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<Float16Type>()
    }

    fn is_bfloat16(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<BFloat16Type>()
    }

    fn is_bool(&self, ctx: &Context) -> bool {
        self.get_type(ctx).deref(ctx).is::<BoolType>()
    }
}

impl<T: Typed> TypedExt for T {}

pub trait TypeExt {
    fn as_ptr(&self, ctx: &Context) -> PointerType;
}

impl TypeExt for TypeHandle {
    fn as_ptr(&self, ctx: &Context) -> PointerType {
        *TypedHandle::from_handle(*self, ctx)
            .expect("Should be pointer")
            .deref(ctx)
    }
}

pub trait ValueExt {
    fn replace_all_uses_except_with(&self, ctx: &Context, except: Use<Value>, other: &Value);
}

impl ValueExt for Value {
    fn replace_all_uses_except_with(&self, ctx: &Context, except: Use<Value>, other: &Value) {
        self.replace_some_uses_with(ctx, |_, r#use| r#use != &except, other);
    }
}

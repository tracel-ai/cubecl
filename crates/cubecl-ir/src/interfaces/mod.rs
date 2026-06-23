use crate::{
    ConstantValue, StorageType,
    dialect::synchronization::SyncScope,
    prelude::*,
    types::{AtomicType, PointerType, VectorType, scalar::*},
};
use pliron::{
    alloc::vec::Vec,
    attribute::AttrObj,
    builtin::attr_interfaces::TypedAttrInterface,
    context::Context,
    derive::{op_interface, type_interface},
    r#type::{TypeHandle, type_cast},
};

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

#[op_interface]
pub trait Pure {
    verify_op_succ!();
}

#[op_interface]
pub trait RematerializeOp {
    verify_op_succ!();
    fn rematerialize(
        &self,
        ctx: &mut Context,
        result_ty: Vec<TypeHandle>,
        operands: Vec<Value>,
    ) -> Ptr<Operation>;
}

macro_rules! rematerialize {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl crate::interfaces::RematerializeOp for $ty {
            fn rematerialize(
                &self,
                ctx: &mut Context,
                result_ty: Vec<TypeHandle>,
                operands: Vec<Value>,
            ) -> Ptr<Operation> {
                Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    result_ty,
                    operands,
                    vec![],
                    0,
                )
            }
        }
    };
}
pub(crate) use rematerialize;

#[op_interface]
pub trait Synchronizes {
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

#[type_interface]
pub trait AlignedType {
    verify_ty_succ!();

    fn align(&self, ctx: &Context) -> usize;
}

macro_rules! aligned {
    ($ty: ty, $align: expr) => {
        #[::pliron::derive::type_interface_impl]
        impl crate::interfaces::AlignedType for $ty {
            #[allow(unused_variables)]
            fn align(&self, ctx: &::pliron::context::Context) -> usize {
                $align
            }
        }
    };
}
pub(crate) use aligned;

#[type_interface]
pub trait SizedType: AlignedType {
    verify_ty_succ!();

    fn size(&self, ctx: &Context) -> usize;
}

macro_rules! sized {
    ($ty: ty, $size: expr) => {
        #[::pliron::derive::type_interface_impl]
        impl crate::interfaces::SizedType for $ty {
            #[allow(unused_variables)]
            fn size(&self, ctx: &::pliron::context::Context) -> usize {
                $size
            }
        }
    };
}
pub(crate) use sized;

macro_rules! erasable {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl pliron::opts::dce::SideEffects for $ty {
            fn has_side_effects(&self, _ctx: &Context) -> bool {
                false
            }
        }
    };
}
pub(crate) use erasable;

#[type_interface]
pub trait MaybeVectorizedType {
    verify_ty_succ!();

    fn vector_size(&self, ctx: &Context) -> usize;
}

macro_rules! scalar {
    ($ty: ty) => {
        #[::pliron::derive::type_interface_impl]
        impl crate::interfaces::MaybeVectorizedType for $ty {
            fn vector_size(&self, _ctx: &::pliron::context::Context) -> usize {
                1
            }
        }
    };
}
pub(crate) use scalar;

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

    fn storage_type(&self, ctx: &Context) -> StorageType;
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

#[op_interface]
pub trait ReadsMemory {
    verify_op_succ!();

    fn reads_through_values(&self, ctx: &Context) -> Vec<Value>;
}

#[op_interface]
pub trait WritesMemory {
    verify_op_succ!();

    fn writes_through_values(&self, ctx: &Context) -> Vec<Value>;
}

#[op_interface]
pub trait SimplifyInterface {
    verify_op_succ!();
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Option<Value>;
}

#[attr_interface]
pub trait ConstantAttr: TypedAttrInterface {
    verify_attr_succ!();
    fn as_const_val(&self) -> ConstantValue;
}

pub trait TypedExt: Typed {
    fn size(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let sized = type_cast::<dyn SizedType>(&*ty).expect("Can't get size of non-sized type");
        sized.size(ctx)
    }

    fn align(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let aligned =
            type_cast::<dyn AlignedType>(&*ty).expect("Can't get align of non-aligned type");
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

    fn is_vector(&self, ctx: &Context, size: usize) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<VectorType>()
            .is_some_and(|it| it.vectorization == size)
    }

    fn is_immutable(&self, ctx: &Context) -> bool {
        !self.is_ptr(ctx)
    }

    fn vector_size(&self, ctx: &Context) -> usize {
        self.try_get_vector_size(ctx)
            .expect("Can't get vector size of non-vectorizable type")
    }

    fn try_get_vector_size(&self, ctx: &Context) -> Option<usize> {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = type_cast::<dyn MaybeVectorizedType>(&*ty)?;
        Some(maybe_vec.vector_size(ctx))
    }

    fn packing_factor(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = type_cast::<dyn MaybeVectorizedType>(&*ty)
            .expect("Can't get vector size of non-vectorizable type");
        maybe_vec.vector_size(ctx)
    }

    fn scalar_ty(&self, ctx: &Context) -> TypeHandle {
        let ty = self.get_type(ctx).deref(ctx);
        let scalarizable = type_cast::<dyn ScalarizableType>(&*ty)
            .expect("Can't get scalar type of non-scalarizable type");
        scalarizable.scalar_type(ctx)
    }

    fn is_int(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.is::<IntType>()
    }

    fn is_int_of_width(&self, ctx: &Context, width: usize) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.downcast_ref::<IntType>()
            .is_some_and(|it| it.width == width)
    }

    fn is_uint(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.is::<UIntType>()
    }

    fn is_uint_of_width(&self, ctx: &Context, width: usize) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.downcast_ref::<UIntType>()
            .is_some_and(|it| it.width == width)
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

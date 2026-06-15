use crate::{
    StorageType,
    dialect::synchronization::SyncScope,
    pliron::prelude::*,
    types::{
        PointerType,
        scalar::{IntType, UIntType},
    },
};
use pliron::{
    derive::{op_interface, type_interface},
    r#type::type_cast,
};

#[macro_export]
macro_rules! verify_op_succ {
    () => {
        fn verify(_op: &dyn pliron::op::Op, _ctx: &Context) -> Result<()>
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
        fn verify(_op: &dyn Type, _ctx: &Context) -> Result<()>
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

    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj>;
}

#[type_interface]
pub trait ScalarType {
    verify_ty_succ!();

    fn storage_type(&self, ctx: &Context) -> StorageType;
}

#[type_interface]
pub trait AggregateType {
    verify_ty_succ!();

    fn field_ty(&self, ctx: &Context, field_idx: usize) -> Ptr<TypeObj>;
}

#[type_interface]
pub trait IndexableType {
    verify_ty_succ!();

    fn indexed_type(&self, ctx: &Context) -> Ptr<TypeObj>;
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

pub trait TypedExt: Typed {
    fn size(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let sized =
            type_cast::<dyn SizedType>(ty.as_ref()).expect("Can't get size of non-sized type");
        sized.size(ctx)
    }

    fn align(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let aligned =
            type_cast::<dyn AlignedType>(ty.as_ref()).expect("Can't get align of non-aligned type");
        aligned.align(ctx)
    }

    fn is_ptr(&self, ctx: &Context) -> bool {
        let ty = self.get_type(ctx).deref(ctx);
        ty.downcast_ref::<PointerType>().is_some()
    }

    fn is_immutable(&self, ctx: &Context) -> bool {
        !self.is_ptr(ctx)
    }

    fn vector_size(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = type_cast::<dyn MaybeVectorizedType>(ty.as_ref())
            .expect("Can't get vector size of non-vectorizable type");
        maybe_vec.vector_size(ctx)
    }

    fn packing_factor(&self, ctx: &Context) -> usize {
        let ty = self.get_type(ctx).deref(ctx);
        let maybe_vec = type_cast::<dyn MaybeVectorizedType>(ty.as_ref())
            .expect("Can't get vector size of non-vectorizable type");
        maybe_vec.vector_size(ctx)
    }

    fn scalar_ty(&self, ctx: &Context) -> Ptr<TypeObj> {
        let ty = self.get_type(ctx).deref(ctx);
        let scalarizable = type_cast::<dyn ScalarizableType>(ty.as_ref())
            .expect("Can't get scalar type of non-scalarizable type");
        scalarizable.scalar_type(ctx)
    }

    fn is_int(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.downcast_ref::<IntType>().is_some()
    }

    fn is_uint(&self, ctx: &Context) -> bool {
        let ty = self.scalar_ty(ctx).deref(ctx);
        ty.downcast_ref::<UIntType>().is_some()
    }
}

impl<T: Typed> TypedExt for T {}

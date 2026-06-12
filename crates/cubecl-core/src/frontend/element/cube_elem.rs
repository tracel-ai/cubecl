use core::fmt::Debug;

use crate::{
    self as cubecl, Assign, IntoRuntime,
    prelude::{Const, CubeDebug, IntoMut, Size},
    unexpanded,
};
use cubecl_ir::{
    ConstantValue, ExpandValue, StorageType,
    features::TypeUsage,
    interfaces::TypedExt,
    pliron::{context::Ptr, r#type::TypeObj},
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use cubecl_runtime::{client::ComputeClient, runtime::Runtime};
use enumset::EnumSet;

use crate::frontend::CubeType;
use crate::ir::Scope;

use super::{NativeAssign, NativeExpand};

/// Form of `CubeType` that encapsulates all primitive types:
/// Numeric, `UInt`, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = NativeExpand<Self>>
    + NativeAssign
    + CubeDebug
    + Send
    + Sync
    + 'static
    + Clone
    + Copy
{
    type Scalar: Scalar;
    type Size: Size;
    type WithScalar<S: Scalar>: CubePrimitive;

    /// Return the element type to use on GPU.
    fn as_type() -> Ptr<TypeObj> {
        unexpanded!()
    }

    /// Only native element types have a size.
    fn size() -> usize {
        size_of::<Self>()
    }

    /// Only native element types have a size.
    fn size_bits() -> usize {
        Self::size() * 8
    }

    fn from_expand_elem(elem: ExpandValue) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(value: ConstantValue) -> Self;

    fn into_lit_unchecked(self) -> Self {
        self
    }

    fn packing_factor() -> usize {
        unexpanded!()
    }

    fn vector_size() -> usize {
        unexpanded!()
    }

    fn __expand_as_type(scope: &Scope) -> Ptr<TypeObj>;

    fn __expand_size(scope: &Scope) -> usize {
        Self::__expand_as_type(scope).size(&scope.ctx())
    }

    fn __expand_size_bits(scope: &Scope) -> usize {
        Self::__expand_size(scope) * 8
    }

    fn __expand_packing_factor(scope: &Scope) -> usize {
        Self::__expand_as_type(scope).packing_factor(&scope.ctx())
    }

    fn __expand_vector_size(scope: &Scope) -> usize {
        Self::__expand_as_type(scope).vector_size(&scope.ctx())
    }
}

pub trait CubePrimitiveExpand {
    type Scalar: Clone + IntoMut + CubeDebug + Assign;
    type WithScalar<S: Scalar>: Clone + IntoMut + CubeDebug + Assign;
}

impl<T: CubePrimitive> CubePrimitiveExpand for NativeExpand<T> {
    type Scalar = NativeExpand<T::Scalar>;
    type WithScalar<S: Scalar> = NativeExpand<T::WithScalar<S>>;
}

/// Marker trait for scalar primitives. Should be implemented for all scalar `CubePrimitive`s, but
/// **not** for `Vector` or non-standard primitives like `Barrier`. Alternatively, treat these as
/// types that can be stored in a [`Vector`]
pub trait Scalar:
    CubePrimitive<Scalar = Self, Size = Const<1>>
    + Default
    + IntoRuntime
    + Debug
    + core::cmp::PartialEq
    + Into<ExpandValue>
{
    fn storage_type(_scope: &Scope) -> StorageType {
        Self::storage_type_native()
    }
    fn storage_type_native() -> StorageType {
        unexpanded!()
    }

    fn supported_uses<R: Runtime>(client: &ComputeClient<R>) -> EnumSet<TypeUsage> {
        let ty = Self::storage_type_native();
        client.features().type_usage(ty)
    }
}

#[cube]
pub fn storage_type_of<E: CubePrimitive>() -> comptime_type!(StorageType) {
    intrinsic!(|scope| { E::Scalar::storage_type(scope) })
}

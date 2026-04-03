use core::fmt::Debug;

use crate::{
    self as cubecl, Assign, IntoRuntime,
    prelude::{Const, CubeDebug, IntoMut, Size},
};
use cubecl_ir::{ConstantValue, ManagedVariable, StorageType, Type, features::TypeUsage};
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
    // + IntoRuntime
    + core::cmp::PartialEq
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
    fn as_type(_scope: &Scope) -> Type {
        Self::as_type_native().expect("To be overridden if not native")
    }

    /// Native or static element type.
    fn as_type_native() -> Option<Type> {
        None
    }

    /// Native or static element type.
    fn as_type_native_unchecked() -> Type {
        Self::as_type_native().expect("To be a native type")
    }

    /// Only native element types have a size.
    fn size() -> Option<usize> {
        Self::as_type_native().map(|t| t.size())
    }

    /// Only native element types have a size.
    fn size_bits() -> Option<usize> {
        Self::as_type_native().map(|t| t.size_bits())
    }

    /// Only native element types have a size.
    fn size_bits_unchecked() -> usize {
        Self::as_type_native_unchecked().size_bits()
    }

    fn from_expand_elem(elem: ManagedVariable) -> Self::ExpandType {
        NativeExpand::new(elem)
    }

    fn from_const_value(value: ConstantValue) -> Self;

    fn into_lit_unchecked(self) -> Self {
        self
    }

    fn supported_uses<R: Runtime>(
        client: &ComputeClient<R>,
    ) -> EnumSet<TypeUsage> {
        let elem = Self::as_type_native_unchecked();
        client.features().type_usage(elem.storage_type())
    }

    fn type_size() -> usize {
        Self::as_type_native_unchecked().size()
    }

    fn type_size_bits() -> usize {
        Self::as_type_native_unchecked().size_bits()
    }

    fn packing_factor() -> usize {
        Self::as_type_native_unchecked().packing_factor()
    }

    fn vector_size() -> usize {
        Self::as_type_native_unchecked().vector_size()
    }

    fn __expand_type_size(scope: &Scope) -> usize {
        Self::as_type(scope).size()
    }

    fn __expand_type_size_bits(scope: &Scope) -> usize {
        Self::as_type(scope).size_bits()
    }

    fn __expand_packing_factor(scope: &Scope) -> usize {
        Self::as_type(scope).packing_factor()
    }

    fn __expand_vector_size(scope: &Scope) -> usize {
        Self::as_type(scope).vector_size()
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
    CubePrimitive<Scalar = Self, Size = Const<1>> + Default + IntoRuntime + Debug
{
}

#[cube]
pub fn type_of<E: CubePrimitive>() -> comptime_type!(Type) {
    intrinsic!(|scope| { E::as_type(scope) })
}

#[cube]
pub fn storage_type_of<E: CubePrimitive>() -> comptime_type!(StorageType) {
    intrinsic!(|scope| { E::as_type(scope).storage_type() })
}

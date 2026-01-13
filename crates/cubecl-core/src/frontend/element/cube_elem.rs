use crate as cubecl;
use cubecl_ir::{ConstantValue, ExpandElement, StorageType, features::TypeUsage};
use cubecl_macros::{comptime_type, cube, intrinsic};
use cubecl_runtime::{client::ComputeClient, runtime::Runtime};
use enumset::EnumSet;

use crate::frontend::CubeType;
use crate::ir::Scope;

use super::{ExpandElementIntoMut, ExpandElementTyped};

/// Form of `CubeType` that encapsulates all primitive types:
/// Numeric, `UInt`, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + ExpandElementIntoMut
    // + IntoRuntime
    + core::cmp::PartialEq
    + Send
    + Sync
    + 'static
    + Clone
    + Copy
{
    /// Return the element type to use on GPU.
    fn as_type(_scope: &Scope) -> StorageType {
        Self::as_type_native().expect("To be overridden if not native")
    }

    /// Native or static element type.
    fn as_type_native() -> Option<StorageType> {
        None
    }

    /// Native or static element type.
    fn as_type_native_unchecked() -> StorageType {
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

    fn from_expand_elem(elem: ExpandElement) -> Self::ExpandType {
        ExpandElementTyped::new(elem)
    }

    fn from_const_value(value: ConstantValue) -> Self;

    fn into_lit_unchecked(self) -> Self {
        self
    }

    fn supported_uses<R: Runtime>(
        client: &ComputeClient<R>,
    ) -> EnumSet<TypeUsage> {
        let elem = Self::as_type_native_unchecked();
        client.properties().features.type_usage(elem)
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

    fn __expand_type_size(scope: &Scope) -> usize {
        Self::as_type(scope).size()
    }

    fn __expand_type_size_bits(scope: &Scope) -> usize {
        Self::as_type(scope).size_bits()
    }

    fn __expand_packing_factor(scope: &Scope) -> usize {
        Self::as_type(scope).packing_factor()
    }
}

#[cube]
pub fn type_of<E: CubePrimitive>() -> comptime_type!(StorageType) {
    intrinsic!(|scope| { E::as_type(scope) })
}

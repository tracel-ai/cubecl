use cubecl_ir::{ConstantScalarValue, ExpandElement, StorageType};
use cubecl_runtime::{TypeUsage, client::ComputeClient, server::ComputeServer};
use enumset::EnumSet;

use crate::frontend::CubeType;
use crate::ir::Scope;

use super::{ExpandElementIntoMut, ExpandElementTyped};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
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

    fn from_const_value(value: ConstantScalarValue) -> Self;

    fn into_lit_unchecked(self) -> Self {
        self
    }

    fn supported_uses<S: ComputeServer>(
        client: &ComputeClient<S>,
    ) -> EnumSet<TypeUsage> {
        let elem = Self::as_type_native_unchecked();
        client.properties().features.type_usage(elem)
    }

    fn elem_size() -> u32 {
        Self::as_type_native_unchecked().size() as u32
    }

    fn elem_size_bits() -> u32 {
        Self::as_type_native_unchecked().size_bits() as u32
    }

    fn packing_factor() -> u32 {
        Self::as_type_native_unchecked().packing_factor()
    }

    fn __expand_elem_size(scope: &Scope) -> u32 {
        Self::as_type(scope).size() as u32
    }

    fn __expand_elem_size_bits(scope: &Scope) -> u32 {
        Self::as_type(scope).size_bits() as u32
    }

    fn __expand_packing_factor(scope: &Scope) -> u32 {
        Self::as_type(scope).packing_factor()
    }
}

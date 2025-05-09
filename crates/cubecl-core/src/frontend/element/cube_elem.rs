use cubecl_ir::ExpandElement;
use cubecl_runtime::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use crate::ir::{Elem, Scope};
use crate::{Feature, frontend::CubeType};

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
    fn as_elem(_scope: &Scope) -> Elem {
        Self::as_elem_native().expect("To be overridden if not native")
    }

    /// Native or static element type.
    fn as_elem_native() -> Option<Elem> {
        None
    }

    /// Native or static element type.
    fn as_elem_native_unchecked() -> Elem {
        Self::as_elem_native().expect("To be a native type")
    }

    /// Only native element types have a size.
    fn size() -> Option<usize> {
        Self::as_elem_native().map(|t| t.size())
    }

    /// Only native element types have a size.
    fn size_bits() -> Option<usize> {
        Self::as_elem_native().map(|t| t.size_bits())
    }

    /// Minimum line size of this type
    fn min_line_size(&self) -> Option<u8> {
        Self::as_elem_native().map(|t| t.min_line_size())
    }


    fn from_expand_elem(elem: ExpandElement) -> Self::ExpandType {
        ExpandElementTyped::new(elem)
    }

    fn is_supported<S: ComputeServer<Feature = Feature>, C: ComputeChannel<S>>(
        client: &ComputeClient<S, C>,
    ) -> bool {
        let elem = Self::as_elem_native_unchecked();
        client.properties().feature_enabled(Feature::Type(elem))
    }

    fn elem_size() -> u32 {
        Self::as_elem_native_unchecked().size() as u32
    }

    fn elem_size_bits() -> u32 {
        Self::as_elem_native_unchecked().size_bits() as u32
    }

    fn __expand_elem_size(scope: &Scope) -> u32 {
        Self::as_elem(scope).size() as u32
    }

    fn __expand_elem_size_bits(scope: &Scope) -> u32 {
        Self::as_elem(scope).size_bits() as u32
    }
}

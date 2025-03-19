use cubecl_ir::ExpandElement;
use cubecl_runtime::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

use crate::ir::{Elem, Scope};
use crate::{Feature, frontend::CubeType};

use super::{ExpandElementBaseInit, ExpandElementTyped};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + ExpandElementBaseInit
    // + IntoRuntime
    + core::cmp::PartialEq
    + Send
    + Sync
    + 'static
    + Clone
    + Copy
{
    /// Return the element type to use on GPU.
    fn as_elem(_context: &Scope) -> Elem {
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

    fn __expand_elem_size(context: &Scope) -> u32 {
        Self::as_elem(context).size() as u32
    }
}

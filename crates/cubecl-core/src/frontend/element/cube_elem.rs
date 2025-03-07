use cubecl_ir::ExpandElement;

use crate::frontend::CubeType;
use crate::ir::{Elem, Scope};

use super::{ExpandElementBaseInit, ExpandElementTyped, IntoRuntime};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + ExpandElementBaseInit
    + IntoRuntime
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
}

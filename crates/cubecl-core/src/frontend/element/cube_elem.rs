use half::{bf16, f16};

use crate::frontend::{CubeType, ExpandElement};
use crate::ir::{Elem, Variable};
use crate::prelude::CubeContext;

use super::{flex32, tf32, ExpandElementBaseInit, ExpandElementTyped, IntoRuntime};

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
    /// Return the element type to use on GPU
    fn as_elem(_context: &CubeContext) -> Elem {
        Self::as_elem_native().expect("To be overriden if not native")
    }

    fn as_elem_native() -> Option<Elem> {
        None
    }

    fn size() -> Option<usize> {
        Self::as_elem_native().map(|t| t.size())
    }

    fn from_expand_elem(elem: ExpandElement) -> Self::ExpandType {
        ExpandElementTyped::new(elem)
    }
}

macro_rules! impl_into_expand_element {
    ($type:ty) => {
        impl From<$type> for ExpandElement {
            fn from(value: $type) -> Self {
                ExpandElement::Plain(Variable::from(value))
            }
        }
    };
}

impl_into_expand_element!(u8);
impl_into_expand_element!(u16);
impl_into_expand_element!(u32);
impl_into_expand_element!(u64);
impl_into_expand_element!(usize);
impl_into_expand_element!(bool);
impl_into_expand_element!(flex32);
impl_into_expand_element!(f16);
impl_into_expand_element!(bf16);
impl_into_expand_element!(tf32);
impl_into_expand_element!(f32);
impl_into_expand_element!(i8);
impl_into_expand_element!(i16);
impl_into_expand_element!(i32);
impl_into_expand_element!(i64);

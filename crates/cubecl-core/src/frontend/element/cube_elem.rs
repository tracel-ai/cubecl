use crate::frontend::UInt;
use crate::frontend::{CubeType, ExpandElement};
use crate::ir::{Elem, Variable};

use super::{AtomicI32, AtomicI64, AtomicUInt, ExpandElementTyped, Vectorized};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + Vectorized
    + core::cmp::Eq
    + core::cmp::PartialEq
    + Send
    + Sync
    + 'static
    + Clone
    + Copy
{
    /// Return the element type to use on GPU
    fn as_elem() -> Elem;

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

impl_into_expand_element!(u32);
impl_into_expand_element!(usize);
impl_into_expand_element!(bool);
impl_into_expand_element!(f32);
impl_into_expand_element!(i32);
impl_into_expand_element!(i64);

/// Useful for Comptime
impl From<UInt> for ExpandElement {
    fn from(value: UInt) -> Self {
        ExpandElement::Plain(crate::ir::Variable::ConstantScalar(
            crate::ir::ConstantScalarValue::UInt(value.val as u64),
        ))
    }
}

impl From<AtomicI32> for ExpandElement {
    fn from(_value: AtomicI32) -> Self {
        todo!()
    }
}

impl From<AtomicI64> for ExpandElement {
    fn from(_value: AtomicI64) -> Self {
        todo!()
    }
}

impl From<AtomicUInt> for ExpandElement {
    fn from(_value: AtomicUInt) -> Self {
        todo!()
    }
}

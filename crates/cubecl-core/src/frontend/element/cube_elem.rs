use crate::compute::KernelLauncher;
use crate::frontend::UInt;
use crate::frontend::{CubeType, ExpandElement};
use crate::ir::{Elem, Variable};
use crate::Runtime;

use super::{ArgSettings, ExpandElementTyped, LaunchArg, LaunchArgExpand, Vectorized};

/// Form of CubeType that encapsulates all primitive types:
/// Numeric, UInt, Bool
pub trait CubePrimitive:
    CubeType<ExpandType = ExpandElementTyped<Self>>
    + LaunchArgExpand
    + Vectorized
    + core::cmp::Eq
    + core::cmp::PartialEq
    + Send
    + Sync
    + 'static
    + Clone
    + Copy
{
    type Primitive: ScalarArgSettings;

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

impl<T: CubePrimitive> LaunchArg for T {
    type RuntimeArg<'a, R: Runtime> = ScalarArg<T>;
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

/// Similar to [ArgSettings], however only for scalar types that don't depend on the [Runtime]
/// trait.
pub trait ScalarArgSettings: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register<R: Runtime>(&self, launcher: &mut KernelLauncher<R>);
}

#[derive(new)]
pub struct ScalarArg<T: CubePrimitive> {
    elem: T::Primitive,
}

impl<T: CubePrimitive, R: Runtime> ArgSettings<R> for ScalarArg<T> {
    fn register(&self, launcher: &mut crate::compute::KernelLauncher<R>) {
        self.elem.register(launcher);
    }
}

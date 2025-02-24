use crate::frontend::CubeType;
use crate::prelude::CubeDebug;
use crate::unexpanded;
use cubecl_ir::Scope;

impl<T: CubeType> CubeType for Option<T>
where
    T::ExpandType: Clone,
{
    type ExpandType = Option<T::ExpandType>;
}

impl<T: CubeDebug> CubeDebug for Option<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        if let Option::Some(value) = &self {
            value.set_debug_name(scope, name)
        }
    }
}

/// Allows to construct a `Some(T)` inside a kernel.
pub fn some<T: CubeType>(_t: T) -> Option<T> {
    unexpanded!()
}

/// Expand of [some].
pub mod some {
    use super::*;

    pub fn expand<T: CubeType>(_context: &mut Scope, t: T::ExpandType) -> Option<T::ExpandType> {
        Some(t)
    }
}

/// Allows to construct a `None` inside a kernel.
pub fn none<T: CubeType>(_t: T) -> Option<T> {
    unexpanded!()
}

/// Expand of [none].
pub mod none {
    use super::*;

    pub fn expand<T: CubeType>(_context: &mut Scope) -> Option<T::ExpandType> {
        None
    }
}

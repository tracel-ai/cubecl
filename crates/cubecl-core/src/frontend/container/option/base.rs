use crate::frontend::CubeType;
use crate::prelude::{CubeDebug, IntoRuntime};
use crate::unexpanded;
use cubecl_ir::Scope;

impl<T: CubeType> CubeType for Option<T> {
    type ExpandType = Option<T::ExpandType>;
}

impl<T: CubeDebug> CubeDebug for Option<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        if let Option::Some(value) = &self {
            value.set_debug_name(scope, name)
        }
    }
}

impl<T: IntoRuntime> IntoRuntime for Option<T> {
    /// Make sure a type is actually expanded into its runtime [expand type](CubeType::ExpandType).
    fn runtime(self) -> Self {
        self
    }

    fn __expand_runtime_method(self, scope: &mut Scope) -> Option<T::ExpandType> {
        self.map(|t| t.__expand_runtime_method(scope))
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
pub fn none<T: CubeType>() -> Option<T> {
    unexpanded!()
}

/// Expand of [none].
pub mod none {
    use super::*;

    pub fn expand<T: CubeType>(_context: &mut Scope) -> Option<T::ExpandType> {
        None
    }
}

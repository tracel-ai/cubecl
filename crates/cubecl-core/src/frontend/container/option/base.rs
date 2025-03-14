use crate::frontend::CubeType;
use crate::prelude::{CubeDebug, IntoRuntime};
use cubecl_ir::Scope;

impl<T: CubeType> CubeType for Option<T>
where
    T::ExpandType: Clone,
{
    type ExpandType = Option<T::ExpandType>;
}

impl<T: IntoRuntime> IntoRuntime for Option<T> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType {
        self.map(|t| t.__expand_runtime_method(scope))
    }
}

impl<T: CubeDebug> CubeDebug for Option<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        if let Option::Some(value) = &self {
            value.set_debug_name(scope, name)
        }
    }
}

#[allow(non_snake_case)]
pub mod Some {
    use super::*;

    pub fn expand<T: CubeType>(_context: &mut Scope, t: T::ExpandType) -> Option<T::ExpandType> {
        Some(t)
    }
}

#[allow(non_snake_case)]
pub mod None {
    use super::*;

    pub fn expand<T: CubeType>(_context: &mut Scope) -> Option<T::ExpandType> {
        None
    }
}

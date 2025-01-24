use cubecl_ir::Scope;

use super::CubeType;

pub trait OptionExt<T: CubeType> {
    fn __expand_unwrap_or_else_method(
        self,
        _scope: &mut Scope,
        other: impl FnOnce(&mut Scope) -> T::ExpandType,
    ) -> T::ExpandType;

    fn __expand_unwrap_or_method(self, _scope: &mut Scope, other: T::ExpandType) -> T::ExpandType;
}

impl<T: CubeType + Into<T::ExpandType>> OptionExt<T> for Option<T> {
    fn __expand_unwrap_or_else_method(
        self,
        scope: &mut Scope,
        other: impl FnOnce(&mut Scope) -> <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.map(Into::into).unwrap_or_else(|| other(scope))
    }

    fn __expand_unwrap_or_method(
        self,
        _scope: &mut Scope,
        other: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.map(Into::into).unwrap_or(other)
    }
}

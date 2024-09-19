use super::{CubeContext, CubeType};

pub trait OptionExt<T: CubeType> {
    fn __expand_unwrap_or_else_method(
        self,
        _context: &mut CubeContext,
        other: impl FnOnce(&mut CubeContext) -> T::ExpandType,
    ) -> T::ExpandType;

    fn __expand_unwrap_or_method(
        self,
        _context: &mut CubeContext,
        other: T::ExpandType,
    ) -> T::ExpandType;
}

impl<T: CubeType + Into<T::ExpandType>> OptionExt<T> for Option<T> {
    fn __expand_unwrap_or_else_method(
        self,
        context: &mut CubeContext,
        other: impl FnOnce(&mut CubeContext) -> <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.map(Into::into).unwrap_or_else(|| other(context))
    }

    fn __expand_unwrap_or_method(
        self,
        _context: &mut CubeContext,
        other: <T as CubeType>::ExpandType,
    ) -> <T as CubeType>::ExpandType {
        self.map(Into::into).unwrap_or(other)
    }
}

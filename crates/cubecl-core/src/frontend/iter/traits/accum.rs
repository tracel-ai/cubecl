use crate::prelude::*;

/// An iterator item that can be read as a number, so `sum` and `product` accept both `T` and `&T`
/// like [`Sum`](core::iter::Sum) and [`Product`](core::iter::Product).
pub trait IntoNumericExpand<S: Numeric> {
    /// Read the item.
    fn into_numeric_expand(self, scope: &Scope) -> NativeExpand<S>;
}

impl<S: Numeric> IntoNumericExpand<S> for NativeExpand<S> {
    fn into_numeric_expand(self, _scope: &Scope) -> NativeExpand<S> {
        self
    }
}

impl<S: Numeric> IntoNumericExpand<S> for &NativeExpand<S> {
    fn into_numeric_expand(self, scope: &Scope) -> NativeExpand<S> {
        self.__expand_deref_method(scope)
    }
}

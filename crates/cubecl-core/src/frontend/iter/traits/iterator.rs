use crate::frontend::iter::{CopiedExpand, FilterExpand, MapExpand};
use crate::frontend::{break_expand, if_expand};
use crate::prelude::*;

/// An argument of a generic iterator method.
///
/// `#[cube]` passes every argument other than a closure as `(arg).into()`, which can't infer a bare
/// generic parameter. The single `From` impl of this wrapper leaves the compiler with one candidate,
/// which infers the parameter from the argument.
pub struct IterArg<T>(pub(super) T);

impl<T> From<T> for IterArg<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

/// The expand half of [`Iterator`](core::iter::Iterator). Consumers expand a single loop through
/// [`Iterable::expand`].
pub trait IteratorExpand: Iterable {
    /// See [`Iterator::map`](core::iter::Iterator::map).
    fn __expand_map_method<B, F>(self, _scope: &Scope, f: F) -> MapExpand<Self, F>
    where
        F: FnMut(&Scope, Self::Item) -> B,
    {
        MapExpand::new(self, f)
    }

    /// See [`Iterator::filter`](core::iter::Iterator::filter).
    fn __expand_filter_method<P>(self, _scope: &Scope, predicate: P) -> FilterExpand<Self, P>
    where
        P: FnMut(&Scope, &Self::Item) -> NativeExpand<bool>,
    {
        FilterExpand::new(self, predicate)
    }

    /// See [`Iterator::copied`](core::iter::Iterator::copied).
    fn __expand_copied_method<'a, T>(self, _scope: &Scope) -> CopiedExpand<Self>
    where
        T: DerefExpand<Target = T> + 'a,
        Self: Iterable<Item = &'a T>,
    {
        CopiedExpand::new(self)
    }

    /// See [`Iterator::for_each`](core::iter::Iterator::for_each).
    fn __expand_for_each_method<F>(self, scope: &Scope, mut f: F)
    where
        F: FnMut(&Scope, Self::Item),
    {
        self.expand(scope, &mut |scope, item| f(scope, item));
    }

    /// See [`Iterator::fold`](core::iter::Iterator::fold).
    fn __expand_fold_method<B, F>(self, scope: &Scope, init: IterArg<B>, mut f: F) -> B::Expand
    where
        B: IntoExpand<Expand: Assign + IntoMut + Clone>,
        F: FnMut(&Scope, B::Expand, Self::Item) -> B::Expand,
    {
        let mut accum = init.0.into_expand(scope).into_mut(scope);
        self.expand(scope, &mut |scope, item| {
            let next = f(scope, accum.clone(), item);
            accum.__expand_assign_method(scope, next);
        });
        accum
    }

    /// See [`Iterator::sum`](core::iter::Iterator::sum).
    ///
    /// The result must be a concrete type such as `sum::<u32>()`, because a generic `F: Float` or
    /// `I: Int` doesn't implement [`Sum`](core::iter::Sum). Use `fold` for those.
    fn __expand_sum_method<S>(self, scope: &Scope) -> NativeExpand<S>
    where
        S: Numeric,
        Self::Item: IntoNumericExpand<S>,
    {
        let mut accum = S::__expand_from_int(scope, 0i64.into()).into_mut(scope);
        self.expand(scope, &mut |scope, item| {
            let value = item.into_numeric_expand(scope);
            accum.__expand_add_assign_method(scope, value);
        });
        accum
    }

    /// See [`Iterator::product`](core::iter::Iterator::product).
    ///
    /// The result must be a concrete type for the same reason as `sum`.
    fn __expand_product_method<P>(self, scope: &Scope) -> NativeExpand<P>
    where
        P: Numeric,
        Self::Item: IntoNumericExpand<P>,
    {
        let mut accum = P::__expand_from_int(scope, 1i64.into()).into_mut(scope);
        self.expand(scope, &mut |scope, item| {
            let value = item.into_numeric_expand(scope);
            accum.__expand_mul_assign_method(scope, value);
        });
        accum
    }

    /// See [`Iterator::count`](core::iter::Iterator::count).
    fn __expand_count_method(self, scope: &Scope) -> NativeExpand<usize> {
        let mut count = NativeExpand::<usize>::from_lit(scope, 0).into_mut(scope);
        self.expand(scope, &mut |scope, _| {
            let one = NativeExpand::<usize>::from_lit(scope, 1);
            count.__expand_add_assign_method(scope, one);
        });
        count
    }

    /// See [`Iterator::any`](core::iter::Iterator::any). Stops at the first match.
    fn __expand_any_method<F>(self, scope: &Scope, mut f: F) -> NativeExpand<bool>
    where
        F: FnMut(&Scope, Self::Item) -> NativeExpand<bool>,
    {
        let mut found = false.__expand_runtime_method(scope).into_mut(scope);
        self.expand(scope, &mut |scope, item| {
            let matched = f(scope, item);
            if_expand(scope, matched, |scope| {
                found.__expand_assign_method(scope, true.into());
                break_expand(scope);
            });
        });
        found
    }

    /// See [`Iterator::all`](core::iter::Iterator::all). Stops at the first item that doesn't match.
    fn __expand_all_method<F>(self, scope: &Scope, mut f: F) -> NativeExpand<bool>
    where
        F: FnMut(&Scope, Self::Item) -> NativeExpand<bool>,
    {
        self.__expand_any_method(scope, |scope, item| {
            f(scope, item).__expand_not_method(scope)
        })
        .__expand_not_method(scope)
    }
}

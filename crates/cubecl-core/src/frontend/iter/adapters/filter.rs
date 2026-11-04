use crate::prelude::*;

/// The expand half of [`Filter`](core::iter::Filter).
#[derive(new)]
pub struct FilterExpand<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> Iterable for FilterExpand<I, P>
where
    I: Iterable,
    P: FnMut(&Scope, &I::Item) -> NativeExpand<bool>,
{
    type Item = I::Item;

    fn expand(mut self, scope: &Scope, body: &mut dyn FnMut(&Scope, I::Item)) {
        self.iter.expand(scope, &mut |scope, item| {
            let keep = (self.predicate)(scope, &item);
            if_expand(scope, keep, |scope| body(scope, item));
        });
    }

    fn expand_unroll(mut self, scope: &Scope, body: &mut dyn FnMut(&Scope, I::Item)) {
        self.iter.expand_unroll(scope, &mut |scope, item| {
            let keep = (self.predicate)(scope, &item);
            if_expand(scope, keep, |scope| body(scope, item));
        });
    }
}

impl<I, P> IteratorExpand for FilterExpand<I, P>
where
    I: IteratorExpand,
    P: FnMut(&Scope, &I::Item) -> NativeExpand<bool>,
{
}

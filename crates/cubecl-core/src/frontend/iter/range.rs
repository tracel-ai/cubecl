use cubecl_ir::ExpandValue;

use crate::prelude::*;

impl<I: Int> IteratorExpand for RangeExpand<I> {}

impl<I: Int> IndexedIteratorExpand for RangeExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        range_len_expand(scope, self.start, self.end)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> NativeExpand<I> {
        range_get_expand(scope, self.start, index)
    }
}

impl<I: Int> IteratorExpand for RangeInclusiveExpand<I> {}

impl<I: Int> IndexedIteratorExpand for RangeInclusiveExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        // `last + 1` would wrap when `last` is the largest value.
        let not_empty = self.start.__expand_le_method(scope, &self.last);
        let len = range_len_expand(scope, self.start, self.last)
            .__expand_add_method(scope, NativeExpand::from_lit(scope, 1));
        select::expand(scope, not_empty, len, NativeExpand::from_lit(scope, 0))
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> NativeExpand<I> {
        range_get_expand(scope, self.start, index)
    }
}

impl<I: Int + Into<ExpandValue>> IteratorExpand for SteppedRangeExpand<I> {}

/// The number of items in `start..end`, which is empty rather than negative when `start > end`.
fn range_len_expand<I: Int>(
    scope: &Scope,
    start: NativeExpand<I>,
    end: NativeExpand<I>,
) -> NativeExpand<usize> {
    let len = end
        .__expand_max_method(scope, start)
        .__expand_sub_method(scope, start);
    usize::__expand_cast_from(scope, len)
}

fn range_get_expand<I: Int>(
    scope: &Scope,
    start: NativeExpand<I>,
    index: NativeExpand<usize>,
) -> NativeExpand<I> {
    let index = I::__expand_cast_from(scope, index);
    start.__expand_add_method(scope, index)
}

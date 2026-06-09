use alloc::boxed::Box;

use crate::{
    self as cubecl,
    ir::{Branch, RangeLoop, Scope},
    prelude::*,
};

#[cube]
pub trait SizedContainer<I: CubePrimitive> {
    /// Return the length of the container.
    fn len(&self) -> I;
}

impl<T: CubeIndex<usize> + SizedContainer<usize> + CubeType<ExpandType = NativeExpand<T>>> Iterable
    for NativeExpand<T>
where
    <T::Output as CubeType>::ExpandType: DerefExpand<Target = <T::Output as CubeType>::ExpandType>,
{
    type Item = <T::Output as CubeType>::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = u32::__expand_as_type(scope);
        let len = self.__expand_len_method(scope);

        let mut child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = NativeExpand::new(i);
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&mut child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0u32.into(),
            end: len.expand,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll array iterator")
    }
}

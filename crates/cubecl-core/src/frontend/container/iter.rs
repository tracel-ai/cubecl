use cubecl_ir::{OpInserter, dialect::branch::RangeLoopOp, types::scalar::IndexType};

use crate::{self as cubecl, ir::Scope, prelude::*};

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
        let index_ty = IndexType::get(scope.ctx());
        let len = self.__expand_len_method(scope);

        let start = scope.const_usize(0);
        let end = len.read_value(scope);
        let step = scope.const_usize(1);

        let i = scope.create_local_mut(index_ty, None);
        let range_loop = RangeLoopOp::new(scope.ctx_mut(), i, start, end, step);
        let loop_body = range_loop.loop_body(scope.ctx());

        let mut child = scope.loop_child(OpInserter::new_at_block_end(loop_body));

        let index = NativeExpand::new(i.into());
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&mut child, item);
        child.terminate_yield();

        register_range_loop::<usize>(scope, &range_loop, &child);
        scope.set_may_return(&[child]);
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll array iterator")
    }
}

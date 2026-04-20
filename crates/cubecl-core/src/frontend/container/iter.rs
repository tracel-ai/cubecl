use alloc::boxed::Box;

use cubecl_ir::Variable;

use crate::{
    ir::{Branch, RangeLoop, Scope},
    prelude::{
        CubeIndex, CubeIndexExpand, CubePrimitive, CubeType, ExpandDeref, Iterable, NativeExpand,
    },
};

use super::Array;

pub trait SizedContainer: CubeIndex<Idx: CubePrimitive, Output = Self::Item> + Sized {
    type Item: CubePrimitive;

    /// Return the length of the container.
    fn len(val: &Variable, scope: &Scope) -> Variable {
        // By default we use the expand len method of the Array type.
        let val: NativeExpand<Array<Self::Item>> = (*val).into();
        val.__expand_len_method(scope).expand
    }
}

impl<T: SizedContainer + CubeType<ExpandType = NativeExpand<T>>> Iterable for NativeExpand<T> {
    type Item = <T::Item as CubeType>::ExpandType;

    fn expand(
        self,
        scope: &Scope,
        mut body: impl FnMut(&Scope, <T::Item as CubeType>::ExpandType),
    ) {
        let index_ty = u32::as_type(scope);
        let len: Variable = T::len(&self.expand, scope);

        let mut child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&mut child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0u32.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(
        self,
        _scope: &Scope,
        _body: impl FnMut(&Scope, <T::Item as CubeType>::ExpandType),
    ) {
        unimplemented!("Can't unroll array iterator")
    }
}

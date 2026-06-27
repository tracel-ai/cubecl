use core::ops::{Index, IndexMut};

use cubecl_ir::{
    ExpandValue, Scope,
    dialect::memory::IndexOp,
    pliron::{builtin::op_interfaces::OneResultInterface, value::Value},
};

use crate::frontend::ReadValue;

use super::{CubeType, NativeExpand, index_expand};

/// Trait bound that can be used to guarantee the expand also implements `IndexExpand`
pub trait CubeIndex<I: CubeType>:
    Index<I, Output: CubeType>
    + CubeType<
        ExpandType: IndexExpand<I::ExpandType, Output = <Self::Output as CubeType>::ExpandType>,
    >
{
    fn __expand_index<'this>(
        scope: &Scope,
        this: &'this Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this <Self::Output as CubeType>::ExpandType {
        this.__expand_index_method(scope, index)
    }
}

impl<I: CubeType, T: Index<I> + CubeType + ?Sized> CubeIndex<I> for T
where
    T::Output: CubeType,
    T::ExpandType:
        IndexExpand<I::ExpandType, Output = <<T as Index<I>>::Output as CubeType>::ExpandType>,
{
}

pub trait IndexExpand<I> {
    type Output;
    fn __expand_index_method(&self, scope: &Scope, index: I) -> &Self::Output;
}

pub trait CubeIndexMut<I: CubeType>:
    CubeIndex<I>
    + IndexMut<I>
    + CubeType<
        ExpandType: IndexMutExpand<I::ExpandType, Output = <Self::Output as CubeType>::ExpandType>,
    >
{
    fn __expand_index_mut<'this>(
        scope: &Scope,
        this: &'this mut Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this mut <Self::Output as CubeType>::ExpandType {
        this.__expand_index_mut_method(scope, index)
    }
}

pub trait IndexMutExpand<I>: IndexExpand<I> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: I,
    ) -> &mut <Self as IndexExpand<I>>::Output;
}

impl<I: CubeType, T: IndexMut<I> + CubeIndex<I> + ?Sized> CubeIndexMut<I> for T
where
    T::Output: CubeType,
    T::ExpandType:
        IndexMutExpand<I::ExpandType, Output = <<T as Index<I>>::Output as CubeType>::ExpandType>,
{
}

pub(crate) fn expand_index_native<'a, O>(
    scope: &Scope,
    list: Value,
    index: NativeExpand<usize>,
    checked: bool,
) -> &'a O
where
    O: From<ExpandValue> + 'static,
{
    let index = index.read_value(scope);
    let val: ExpandValue = index_expand(scope, list, index, checked).into();

    scope.create_kernel_ref(val.into())
}

pub(crate) fn expand_index_mut_native<'a, O>(
    scope: &Scope,
    list: Value,
    index: NativeExpand<usize>,
    checked: bool,
) -> &'a mut O
where
    O: From<ExpandValue> + 'static,
{
    let index = index.read_value(scope);

    let index_op = IndexOp::maybe_checked(scope.ctx_mut(), list, index, checked);
    scope.register(&index_op);
    let out: ExpandValue = index_op.get_result(scope.ctx()).into();

    scope.create_kernel_ref(out.into())
}

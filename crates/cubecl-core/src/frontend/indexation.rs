use core::ops::{Index, IndexMut};

use cubecl_ir::{IndexOperands, Instruction, Memory, Scope, Type, Value, ValueKind};

use super::{CubeType, NativeExpand, index_expand};
use crate::prelude::CubePrimitive;

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
    O: From<Value> + 'static,
{
    let index: Value = index.into();
    let index_var: Value = index;
    let index = match index_var.kind {
        ValueKind::Constant(value) => Value::constant(value, usize::__expand_as_type(scope)),
        _ => index,
    };
    let val = index_expand(scope, list, index, checked);

    scope.create_kernel_ref(val.into())
}

pub(crate) fn expand_index_mut_native<'a, O>(
    scope: &Scope,
    list: Value,
    index: NativeExpand<usize>,
    checked: bool,
) -> &'a mut O
where
    O: From<Value> + 'static,
{
    let index: Value = index.expand;
    let index = match index.kind {
        ValueKind::Constant(value) => Value::constant(value, usize::__expand_as_type(scope)),
        _ => index,
    };

    let ty = list.value_type();
    let class = list.address_space();
    let out = scope.create_value(Type::pointer(ty, class));

    scope.register(Instruction::new(
        Memory::Index(IndexOperands {
            list,
            index,
            unroll_factor: 1,
            checked,
        }),
        out,
    ));

    scope.create_kernel_ref(out.into())
}

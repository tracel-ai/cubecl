use core::ops::{Index, IndexMut};

use cubecl_ir::{
    IndexOperator, Instruction, Memory, Scope, Type, Variable, VariableKind, VectorSize,
};

use super::{CubeType, NativeExpand, index_expand};
use crate::{frontend::ExpandTypeClone, prelude::CubePrimitive};

/// Trait bound that can be used to guarantee the expand also implements `IndexExpand`
pub trait CubeIndex<I: CubeType>:
    Index<I, Output: CubeType>
    + CubeType<
        ExpandType: IndexExpand<I::ExpandType, Output = <Self::Output as CubeType>::ExpandType>,
    >
{
    fn index_unchecked(&self, index: I) -> &Self::Output {
        self.index(index)
    }

    fn __expand_index<'this>(
        scope: &Scope,
        this: &'this Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this <Self::Output as CubeType>::ExpandType {
        this.__expand_index_method(scope, index)
    }
    fn __expand_index_unchecked<'this>(
        scope: &Scope,
        this: &'this Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this <Self::Output as CubeType>::ExpandType {
        this.__expand_index_unchecked_method(scope, index)
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
    fn __expand_index_unchecked_method(&self, scope: &Scope, index: I) -> &Self::Output;
}

pub trait CubeIndexMut<I: CubeType>:
    CubeIndex<I>
    + IndexMut<I>
    + CubeType<
        ExpandType: IndexMutExpand<I::ExpandType, Output = <Self::Output as CubeType>::ExpandType>,
    >
{
    fn index_mut_unchecked(&mut self, index: I) -> &mut Self::Output {
        self.index_mut(index)
    }

    fn __expand_index_mut<'this>(
        scope: &Scope,
        this: &'this mut Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this mut <Self::Output as CubeType>::ExpandType {
        this.__expand_index_mut_method(scope, index)
    }
    fn __expand_index_mut_unchecked<'this>(
        scope: &Scope,
        this: &'this mut Self::ExpandType,
        index: I::ExpandType,
    ) -> &'this mut <Self::Output as CubeType>::ExpandType {
        this.__expand_index_mut_unchecked_method(scope, index)
    }
}

pub trait IndexMutExpand<I>: IndexExpand<I> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: I,
    ) -> &mut <Self as IndexExpand<I>>::Output;
    fn __expand_index_mut_unchecked_method(
        &mut self,
        scope: &Scope,
        index: I,
    ) -> &mut <Self as IndexExpand<I>>::Output;
}

impl<I: CubeType, T: IndexMut<I> + CubeIndex<I>> CubeIndexMut<I> for T
where
    T::Output: CubeType,
    T::ExpandType:
        IndexMutExpand<I::ExpandType, Output = <<T as Index<I>>::Output as CubeType>::ExpandType>,
{
}

pub(crate) fn expand_index_native<'a, A: ExpandTypeClone + Into<Variable>, O>(
    scope: &Scope,
    array: &'a A,
    index: NativeExpand<usize>,
    vector_size: Option<VectorSize>,
    checked: bool,
) -> &'a O
where
    O: From<Variable> + 'static,
{
    let index: Variable = index.into();
    let index_var: Variable = index;
    let index = match index_var.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::__expand_as_type(scope)),
        _ => index,
    };
    let array: Variable = array.clone_unchecked().into();
    let var = index_expand(scope, array, index, vector_size, checked);

    scope.create_kernel_ref(var.into())
}

pub(crate) fn expand_index_mut_native<'a, A: ExpandTypeClone + Into<Variable>, O>(
    scope: &Scope,
    list: &'a mut A,
    index: NativeExpand<usize>,
    vector_size: Option<VectorSize>,
    checked: bool,
) -> &'a mut O
where
    O: From<Variable> + 'static,
{
    let list: Variable = list.clone_unchecked().into();
    let index: Variable = index.expand;
    let index = match index.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::__expand_as_type(scope)),
        _ => index,
    };

    let ty = match vector_size {
        Some(vector_size) => list.value_type().with_vector_size(vector_size),
        None => list.value_type(),
    };
    let class = list.pointer_class();
    let out = scope.create_local(Type::pointer(ty, class));
    let vector_size = vector_size.unwrap_or(0);

    scope.register(Instruction::new(
        Memory::Index(IndexOperator {
            list,
            index,
            vector_size,
            unroll_factor: 1,
            checked,
        }),
        out,
    ));

    scope.create_kernel_ref(out.into())
}

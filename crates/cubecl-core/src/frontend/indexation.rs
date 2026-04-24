use core::ops::{Index, IndexMut};

use cubecl_ir::{
    IndexOperator, Instruction, Operator, Scope, Type, Variable, VariableKind, VectorSize,
};

use super::{CubeType, NativeExpand, index_expand};
use crate::{prelude::CubePrimitive, unexpanded};

/// Fake indexation so we can rewrite indexes into scalars as calls to this fake function in the
/// non-expanded function
pub trait CubeIndex:
    CubeType<
    ExpandType: CubeIndexExpand<
        Idx = <Self::Idx as CubeType>::ExpandType,
        Output = <Self::Output as CubeType>::ExpandType,
    >,
>
{
    type Output: CubeType;
    type Idx: CubeType;

    fn cube_idx(&self, _i: Self::Idx) -> &Self::Output {
        unexpanded!()
    }
}

/// Workaround for comptime indexing, since the helper that replaces index operators doesn't know
/// about whether a variable is comptime. Has the same signature in unexpanded code, so it will
/// automatically dispatch the correct one.
pub trait ComptimeIndex<I>: Index<I> {
    fn cube_idx(&self, i: I) -> &Self::Output {
        self.index(i)
    }
}

impl<I, T: Index<I>> ComptimeIndex<I> for T {}
impl<I, T: IndexMut<I>> ComptimeIndexMut<I> for T {}

pub trait ComptimeIndexMut<I>: ComptimeIndex<I> + IndexMut<I> {
    fn cube_idx_mut(&mut self, i: I) -> &mut Self::Output {
        self.index_mut(i)
    }
}

pub trait CubeIndexExpand {
    type Output;
    type Idx;
    fn __expand_index_method(&self, scope: &Scope, index: Self::Idx) -> &Self::Output;
    fn __expand_index_unchecked_method(&self, scope: &Scope, index: Self::Idx) -> &Self::Output;
}

pub trait CubeIndexMut:
    CubeIndex
    + CubeType<ExpandType: CubeIndexMutExpand<Output = <Self::Output as CubeType>::ExpandType>>
{
    fn cube_idx_mut(&mut self, _i: <Self as CubeIndex>::Idx) -> &mut <Self as CubeIndex>::Output {
        unexpanded!()
    }
}

pub trait CubeIndexMutExpand: CubeIndexExpand {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: <Self as CubeIndexExpand>::Idx,
    ) -> &mut <Self as CubeIndexExpand>::Output;
}

pub(crate) fn expand_index_native<'a, A: CubeIndexExpand + Clone + Into<Variable>>(
    scope: &Scope,
    array: &'a A,
    index: NativeExpand<usize>,
    vector_size: Option<VectorSize>,
    checked: bool,
) -> &'a A::Output
where
    A::Output: From<Variable> + 'static,
{
    let index: Variable = index.into();
    let index_var: Variable = index;
    let index = match index_var.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::as_type(scope)),
        _ => index,
    };
    let array: Variable = array.clone().into();
    let var = if checked {
        index_expand(scope, array, index, vector_size, Operator::Index)
    } else {
        index_expand(scope, array, index, vector_size, Operator::UncheckedIndex)
    };

    scope.create_kernel_ref(var.into())
}

pub(crate) fn expand_index_mut_native<'a, A: CubeIndexMutExpand + Clone + Into<Variable>>(
    scope: &Scope,
    list: &'a mut A,
    index: NativeExpand<usize>,
    vector_size: Option<VectorSize>,
    checked: bool,
) -> &'a mut A::Output
where
    A::Output: From<Variable> + 'static,
{
    let list: Variable = list.clone().into();
    let index: Variable = index.expand;
    let index = match index.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::as_type(scope)),
        _ => index,
    };

    let ty = match vector_size {
        Some(vector_size) => list.ty.with_vector_size(vector_size),
        None => list.ty,
    };
    let class = list.pointer_class();
    let out = scope.create_local(Type::pointer(ty, class));
    let vector_size = vector_size.unwrap_or(0);

    if checked {
        scope.register(Instruction::new(
            Operator::IndexMut(IndexOperator {
                list,
                index,
                vector_size,
                unroll_factor: 1,
            }),
            out,
        ));
    } else {
        scope.register(Instruction::new(
            Operator::UncheckedIndexMut(IndexOperator {
                list,
                index,
                vector_size,
                unroll_factor: 1,
            }),
            out,
        ));
    }

    scope.create_kernel_ref(out.into())
}

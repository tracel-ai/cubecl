use core::ops::{Index, IndexMut};

use cubecl_ir::{
    ExpandElement, IndexAssignOperator, Instruction, LineSize, Operator, Scope, VariableKind,
};

use super::{CubeType, ExpandElementTyped, index_expand, index_expand_no_vec};
use crate::{ir::Variable, prelude::CubePrimitive, unexpanded};

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

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as CubeType>::ExpandType,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.expand_index(scope, index)
    }
    fn expand_index_unchecked(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as CubeType>::ExpandType,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.expand_index_unchecked(scope, index)
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
    fn expand_index(self, scope: &mut Scope, index: Self::Idx) -> Self::Output;
    fn expand_index_unchecked(self, scope: &mut Scope, index: Self::Idx) -> Self::Output;
}

pub trait CubeIndexMut:
    CubeIndex
    + CubeType<ExpandType: CubeIndexMutExpand<Output = <Self::Output as CubeType>::ExpandType>>
{
    fn cube_idx_mut(&mut self, _i: Self::Idx) -> &mut <Self as CubeIndex>::Output {
        unexpanded!()
    }
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as CubeType>::ExpandType,
        value: <Self::Output as CubeType>::ExpandType,
    ) {
        array.expand_index_mut(scope, index, value)
    }
}

pub trait CubeIndexMutExpand: CubeIndexExpand {
    fn expand_index_mut(self, scope: &mut Scope, index: Self::Idx, value: Self::Output);
}

pub(crate) fn expand_index_native<A: CubeType + CubeIndex>(
    scope: &mut Scope,
    array: ExpandElementTyped<A>,
    index: ExpandElementTyped<usize>,
    line_size: Option<LineSize>,
    checked: bool,
) -> ExpandElementTyped<A::Output>
where
    A::Output: CubeType + Sized,
{
    let index: ExpandElement = index.into();
    let index_var: Variable = *index;
    let index = match index_var.kind {
        VariableKind::Constant(value) => {
            ExpandElement::Plain(Variable::constant(value, usize::as_type(scope)))
        }
        _ => index,
    };
    let array: ExpandElement = array.into();
    let var: Variable = *array;
    let var = if checked {
        match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } => {
                index_expand_no_vec(scope, array, index, Operator::Index)
            }
            _ => index_expand(scope, array, index, line_size, Operator::Index),
        }
    } else {
        match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } => {
                index_expand_no_vec(scope, array, index, Operator::UncheckedIndex)
            }
            _ => index_expand(scope, array, index, line_size, Operator::UncheckedIndex),
        }
    };

    ExpandElementTyped::new(var)
}

pub(crate) fn expand_index_assign_native<
    A: CubeType<ExpandType = ExpandElementTyped<A>> + CubeIndexMut,
>(
    scope: &mut Scope,
    array: A::ExpandType,
    index: ExpandElementTyped<usize>,
    value: ExpandElementTyped<<A as CubeIndex>::Output>,
    line_size: Option<LineSize>,
    checked: bool,
) where
    A::Output: CubeType + Sized,
{
    let index: Variable = index.expand.into();
    let index = match index.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::as_type(scope)),
        _ => index,
    };

    let line_size = line_size.unwrap_or(0);
    if checked {
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                line_size,
                unroll_factor: 1,
            }),
            array.expand.into(),
        ));
    } else {
        scope.register(Instruction::new(
            Operator::UncheckedIndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                line_size,
                unroll_factor: 1,
            }),
            array.expand.into(),
        ));
    }
}

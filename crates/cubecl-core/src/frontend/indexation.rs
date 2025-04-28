use cubecl_ir::{ExpandElement, IndexAssignOperator, Instruction, Operator, Scope, VariableKind};

use super::{CubeType, ExpandElementTyped, index_expand, index_expand_no_vec};
use crate::{
    ir::{IntKind, UIntKind, Variable},
    unexpanded,
};

/// Fake indexation so we can rewrite indexes into scalars as calls to this fake function in the
/// non-expanded function
pub trait CubeIndex:
    CubeType<ExpandType: CubeIndexExpand<Output = <Self::Output as CubeType>::ExpandType>>
{
    type Output: CubeType;

    fn cube_idx(&self, _i: u32) -> &Self::Output {
        unexpanded!()
    }

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.expand_index(scope, index)
    }
    fn expand_index_unchecked(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.expand_index_unchecked(scope, index)
    }
}

pub trait CubeIndexExpand {
    type Output;
    fn expand_index(self, scope: &mut Scope, index: ExpandElementTyped<u32>) -> Self::Output;
    fn expand_index_unchecked(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> Self::Output;
}

pub trait CubeIndexMut:
    CubeIndex
    + CubeType<ExpandType: CubeIndexMutExpand<Output = <Self::Output as CubeType>::ExpandType>>
{
    fn cube_idx_mut(&mut self, _i: u32) -> &mut Self::Output {
        unexpanded!()
    }
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: <Self::Output as CubeType>::ExpandType,
    ) {
        array.expand_index_mut(scope, index, value)
    }
}

pub trait CubeIndexMutExpand {
    type Output;
    fn expand_index_mut(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: Self::Output,
    );
}

pub(crate) fn expand_index_native<A: CubeType + CubeIndex>(
    scope: &mut Scope,
    array: ExpandElementTyped<A>,
    index: ExpandElementTyped<u32>,
    line_size: Option<u32>,
    checked: bool,
) -> ExpandElementTyped<A::Output>
where
    A::Output: CubeType + Sized,
{
    let index: ExpandElement = index.into();
    let index_var: Variable = *index;
    let index = match index_var.kind {
        VariableKind::ConstantScalar(value) => ExpandElement::Plain(Variable::constant(
            crate::ir::ConstantScalarValue::UInt(value.as_u64(), UIntKind::U32),
        )),
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
    index: ExpandElementTyped<u32>,
    value: ExpandElementTyped<A::Output>,
    line_size: Option<u32>,
    checked: bool,
) where
    A::Output: CubeType + Sized,
{
    let index: Variable = index.expand.into();
    let index = match index.kind {
        VariableKind::ConstantScalar(value) => Variable::constant(
            crate::ir::ConstantScalarValue::UInt(value.as_u64(), UIntKind::U32),
        ),
        _ => index,
    };
    let line_size = line_size.unwrap_or(0);
    if checked {
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                line_size,
            }),
            array.expand.into(),
        ));
    } else {
        scope.register(Instruction::new(
            Operator::UncheckedIndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                line_size,
            }),
            array.expand.into(),
        ));
    }
}

pub trait Index {
    fn value(self) -> Variable;
}

impl Index for i32 {
    fn value(self) -> Variable {
        Variable::constant(crate::ir::ConstantScalarValue::Int(
            self as i64,
            IntKind::I32,
        ))
    }
}

impl Index for u32 {
    fn value(self) -> Variable {
        Variable::constant(crate::ir::ConstantScalarValue::UInt(
            self as u64,
            UIntKind::U32,
        ))
    }
}

impl Index for ExpandElement {
    fn value(self) -> Variable {
        *self
    }
}

impl Index for ExpandElementTyped<u32> {
    fn value(self) -> Variable {
        *self.expand
    }
}

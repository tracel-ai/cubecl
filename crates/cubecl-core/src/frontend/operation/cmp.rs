use crate::frontend::operation::base::cmp_expand;
use crate::frontend::{CubeContext, ExpandElementTyped};
use crate::ir::Operator;
use crate::prelude::CubePrimitive;

pub mod ne {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::NotEqual,
        )
        .into()
    }
}

pub mod gt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::Greater,
        )
        .into()
    }
}

pub mod lt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::Lower,
        )
        .into()
    }
}

pub mod ge {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::GreaterEqual,
        )
        .into()
    }
}

pub mod le {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::LowerEqual,
        )
        .into()
    }
}

pub mod eq {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(
            context,
            lhs.into().into(),
            rhs.into().into(),
            Operator::Equal,
        )
        .into()
    }
}

pub mod add_assign {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: impl Into<ExpandElementTyped<C>>,
        rhs: impl Into<ExpandElementTyped<C>>,
    ) -> ExpandElementTyped<C> {
        cmp_expand(context, lhs.into().into(), rhs.into().into(), Operator::Add).into()
    }
}

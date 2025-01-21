use crate::frontend::operation::base::cmp_expand;
use crate::frontend::{CubeContext, ExpandElementTyped};
use crate::ir::Comparison;
use crate::prelude::CubePrimitive;

pub mod ne {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::NotEqual).into()
    }
}

pub mod gt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::Greater).into()
    }
}

pub mod lt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::Lower).into()
    }
}

pub mod ge {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::GreaterEqual).into()
    }
}

pub mod le {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::LowerEqual).into()
    }
}

pub mod eq {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        context: &mut CubeContext,
        lhs: ExpandElementTyped<C>,
        rhs: ExpandElementTyped<C>,
    ) -> ExpandElementTyped<bool> {
        cmp_expand(context, lhs.into(), rhs.into(), Comparison::Equal).into()
    }
}

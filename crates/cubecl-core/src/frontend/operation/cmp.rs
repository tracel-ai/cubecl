use crate::frontend::NativeExpand;
use crate::frontend::operation::base::cmp_expand;
use crate::ir::{Comparison, Scope};
use crate::prelude::CubePrimitive;

// NOTE: Unary comparison tests are in the unary module

pub mod ne {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::NotEqual).into()
    }
}

pub mod gt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::Greater).into()
    }
}

pub mod lt {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::Lower).into()
    }
}

pub mod ge {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::GreaterEqual).into()
    }
}

pub mod le {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::LowerEqual).into()
    }
}

pub mod eq {

    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        lhs: NativeExpand<C>,
        rhs: NativeExpand<C>,
    ) -> NativeExpand<bool> {
        cmp_expand(scope, lhs.into(), rhs.into(), Comparison::Equal).into()
    }
}

use cubecl_ir::ExpandElement;
use num_traits::NumCast;

use crate::ir::Switch;
use crate::ir::{Branch, If, IfElse, Item, Loop, RangeLoop, Scope};

use super::{assign, CubePrimitive, CubeType, ExpandElementTyped, Int, Numeric};

/// Something that can be iterated on by a for loop. Currently only includes `Range`, `StepBy` and
/// `Sequence`.
pub trait Iterable<T: CubeType>: Sized {
    /// Expand a runtime loop without unrolling
    ///
    /// # Arguments
    /// # Arguments
    /// * `scope` - the expansion scope
    /// * `body` - the loop body to be executed repeatedly
    fn expand(self, scope: &mut Scope, body: impl FnMut(&mut Scope, <T as CubeType>::ExpandType));
    /// Expand an unrolled loop. The body should be invoced `n` times, where `n` is the number of
    /// iterations.
    ///
    /// # Arguments
    /// * `scope` - the expansion scope
    /// * `body` - the loop body to be executed repeatedly
    fn expand_unroll(
        self,
        scope: &mut Scope,
        body: impl FnMut(&mut Scope, <T as CubeType>::ExpandType),
    );
}

pub struct RangeExpand<I: Int> {
    pub start: ExpandElementTyped<I>,
    pub end: ExpandElementTyped<I>,
    pub inclusive: bool,
}

impl<I: Int> RangeExpand<I> {
    pub fn new(start: ExpandElementTyped<I>, end: ExpandElementTyped<I>, inclusive: bool) -> Self {
        RangeExpand {
            start,
            end,
            inclusive,
        }
    }

    pub fn __expand_step_by_method(
        self,
        n: impl Into<ExpandElementTyped<u32>>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start: self.start,
            end: self.end,
            step: n.into(),
            inclusive: self.inclusive,
        }
    }
}

impl<I: Int> Iterable<I> for RangeExpand<I> {
    fn expand_unroll(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <I as CubeType>::ExpandType),
    ) {
        let start = self
            .start
            .expand
            .as_const()
            .expect("Only constant start can be unrolled.")
            .as_i64();
        let end = self
            .end
            .expand
            .as_const()
            .expect("Only constant end can be unrolled.")
            .as_i64();

        if self.inclusive {
            for i in start..=end {
                let var = I::from_int(i);
                body(scope, var.into())
            }
        } else {
            for i in start..end {
                let var = I::from_int(i);
                body(scope, var.into())
            }
        }
    }

    fn expand(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <I as CubeType>::ExpandType),
    ) {
        let mut child = scope.child();
        let index_ty = Item::new(I::as_elem(scope));
        let i = child.create_local_restricted(index_ty);

        body(&mut child, i.clone().into());

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start: *self.start.expand,
            end: *self.end.expand,
            step: None,
            scope: child,
            inclusive: self.inclusive,
        })));
    }
}

pub struct SteppedRangeExpand<I: Int> {
    start: ExpandElementTyped<I>,
    end: ExpandElementTyped<I>,
    step: ExpandElementTyped<u32>,
    inclusive: bool,
}

impl<I: Int + Into<ExpandElement>> Iterable<I> for SteppedRangeExpand<I> {
    fn expand(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <I as CubeType>::ExpandType),
    ) {
        let mut child = scope.child();
        let index_ty = Item::new(I::as_elem(scope));
        let i = child.create_local_restricted(index_ty);

        body(&mut child, i.clone().into());

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start: *self.start.expand,
            end: *self.end.expand,
            step: Some(*self.step.expand),
            scope: child,
            inclusive: self.inclusive,
        })));
    }

    fn expand_unroll(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <I as CubeType>::ExpandType),
    ) {
        let start = self
            .start
            .expand
            .as_const()
            .expect("Only constant start can be unrolled.")
            .as_i64();
        let end = self
            .end
            .expand
            .as_const()
            .expect("Only constant end can be unrolled.")
            .as_i64();
        let step = self
            .step
            .expand
            .as_const()
            .expect("Only constant step can be unrolled.")
            .as_usize();

        if self.inclusive {
            for i in (start..=end).step_by(step) {
                let var = I::from_int(i);
                body(scope, var.into())
            }
        } else {
            for i in (start..end).step_by(step) {
                let var = I::from_int(i);
                body(scope, var.into())
            }
        }
    }
}

/// integer range. Equivalent to:
///
/// ```ignore
/// start..end
/// ```
pub fn range<T: Int>(start: T, end: T) -> impl Iterator<Item = T> {
    let start: i64 = start.to_i64().unwrap();
    let end: i64 = end.to_i64().unwrap();
    (start..end).map(<T as NumCast>::from).map(Option::unwrap)
}

pub mod range {
    use cubecl_ir::Scope;

    use crate::prelude::{ExpandElementTyped, Int};

    use super::RangeExpand;

    pub fn expand<I: Int>(
        _scope: &mut Scope,
        start: ExpandElementTyped<I>,
        end: ExpandElementTyped<I>,
    ) -> RangeExpand<I> {
        RangeExpand {
            start,
            end,
            inclusive: false,
        }
    }
}

/// Stepped range. Equivalent to:
///
/// ```ignore
/// (start..end).step_by(step)
/// ```
///
/// Allows using any integer for the step, instead of just usize
pub fn range_stepped<I: Int>(start: I, end: I, step: impl Int) -> impl Iterator<Item = I> {
    let start = start.to_i64().unwrap();
    let end = end.to_i64().unwrap();
    let step = step.to_usize().unwrap();
    (start..end)
        .step_by(step)
        .map(<I as NumCast>::from)
        .map(Option::unwrap)
}

pub mod range_stepped {
    use cubecl_ir::Scope;

    use crate::prelude::{ExpandElementTyped, Int};

    use super::SteppedRangeExpand;

    pub fn expand<I: Int>(
        _scope: &mut Scope,
        start: ExpandElementTyped<I>,
        end: ExpandElementTyped<I>,
        step: ExpandElementTyped<u32>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start,
            end,
            step,
            inclusive: false,
        }
    }
}

pub fn for_expand<I: Numeric>(
    scope: &mut Scope,
    range: impl Iterable<I>,
    unroll: bool,
    body: impl FnMut(&mut Scope, ExpandElementTyped<I>),
) {
    if unroll {
        range.expand_unroll(scope, body);
    } else {
        range.expand(scope, body);
    }
}

pub fn if_expand(scope: &mut Scope, runtime_cond: ExpandElement, block: impl FnOnce(&mut Scope)) {
    let comptime_cond = runtime_cond.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(cond) => {
            if cond {
                block(scope);
            }
        }
        None => {
            let mut child = scope.child();

            block(&mut child);

            scope.register(Branch::If(Box::new(If {
                cond: *runtime_cond,
                scope: child,
            })));
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum IfElseExpand {
    ComptimeThen,
    ComptimeElse,
    Runtime {
        runtime_cond: ExpandElement,
        then_child: Scope,
    },
}

impl IfElseExpand {
    pub fn or_else(self, scope: &mut Scope, else_block: impl FnOnce(&mut Scope)) {
        match self {
            Self::Runtime {
                runtime_cond,
                then_child,
            } => {
                let mut else_child = scope.child();
                else_block(&mut else_child);

                scope.register(Branch::IfElse(Box::new(IfElse {
                    cond: *runtime_cond,
                    scope_if: then_child,
                    scope_else: else_child,
                })));
            }
            Self::ComptimeElse => else_block(scope),
            Self::ComptimeThen => (),
        }
    }
}

pub fn if_else_expand(
    scope: &mut Scope,
    runtime_cond: ExpandElement,
    then_block: impl FnOnce(&mut Scope),
) -> IfElseExpand {
    let comptime_cond = runtime_cond.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            then_block(scope);
            IfElseExpand::ComptimeThen
        }
        Some(false) => IfElseExpand::ComptimeElse,
        None => {
            let mut then_child = scope.child();
            then_block(&mut then_child);

            IfElseExpand::Runtime {
                runtime_cond,
                then_child,
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum IfElseExprExpand<C: CubeType> {
    ComptimeThen(ExpandElementTyped<C>),
    ComptimeElse,
    Runtime {
        runtime_cond: ExpandElement,
        out: ExpandElementTyped<C>,
        then_child: Scope,
    },
}

impl<C: CubePrimitive> IfElseExprExpand<C> {
    pub fn or_else(
        self,
        scope: &mut Scope,
        else_block: impl FnOnce(&mut Scope) -> ExpandElementTyped<C>,
    ) -> ExpandElementTyped<C> {
        match self {
            Self::Runtime {
                runtime_cond,
                out,
                then_child,
            } => {
                let mut else_child = scope.child();
                let ret = else_block(&mut else_child);
                assign::expand::<C>(&mut else_child, ret, out.clone());

                scope.register(Branch::IfElse(Box::new(IfElse {
                    cond: *runtime_cond,
                    scope_if: then_child,
                    scope_else: else_child,
                })));
                out
            }
            Self::ComptimeElse => else_block(scope),
            Self::ComptimeThen(ret) => ret,
        }
    }
}

pub fn if_else_expr_expand<C: CubePrimitive>(
    scope: &mut Scope,
    runtime_cond: ExpandElement,
    then_block: impl FnOnce(&mut Scope) -> ExpandElementTyped<C>,
) -> IfElseExprExpand<C> {
    let comptime_cond = runtime_cond.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            let ret = then_block(scope);
            IfElseExprExpand::ComptimeThen(ret)
        }
        Some(false) => IfElseExprExpand::ComptimeElse,
        None => {
            let mut then_child = scope.child();
            let ret = then_block(&mut then_child);
            let out: ExpandElementTyped<C> = scope.create_local_mut(ret.expand.item).into();
            assign::expand::<C>(&mut then_child, ret, out.clone());

            IfElseExprExpand::Runtime {
                runtime_cond,
                out,
                then_child,
            }
        }
    }
}

pub struct SwitchExpand<I: Int> {
    value: ExpandElementTyped<I>,
    default: Scope,
    cases: Vec<(ExpandElementTyped<I>, Scope)>,
}

impl<I: Int> SwitchExpand<I> {
    pub fn case(
        mut self,
        scope: &mut Scope,
        value: impl Int,
        block: impl FnOnce(&mut Scope),
    ) -> Self {
        let value = I::from(value).unwrap();
        let mut case_child = scope.child();
        block(&mut case_child);
        self.cases.push((value.into(), case_child));
        self
    }

    pub fn finish(self, scope: &mut Scope) {
        let value_var = *self.value.expand;
        scope.register(Branch::Switch(Box::new(Switch {
            value: value_var,
            scope_default: self.default,
            cases: self
                .cases
                .into_iter()
                .map(|it| (*it.0.expand, it.1))
                .collect(),
        })));
    }
}

pub fn switch_expand<I: Int>(
    scope: &mut Scope,
    value: ExpandElementTyped<I>,
    default_block: impl FnOnce(&mut Scope),
) -> SwitchExpand<I> {
    let mut default_child = scope.child();
    default_block(&mut default_child);

    SwitchExpand {
        value,
        default: default_child,
        cases: Vec::new(),
    }
}

pub struct SwitchExpandExpr<I: Int, C: CubePrimitive> {
    value: ExpandElementTyped<I>,
    out: ExpandElementTyped<C>,
    default: Scope,
    cases: Vec<(ExpandElementTyped<I>, Scope)>,
}

impl<I: Int, C: CubePrimitive> SwitchExpandExpr<I, C> {
    pub fn case(
        mut self,
        scope: &mut Scope,
        value: impl Int,
        block: impl FnOnce(&mut Scope) -> ExpandElementTyped<C>,
    ) -> Self {
        let value = I::from(value).unwrap();
        let mut case_child = scope.child();
        let ret = block(&mut case_child);
        assign::expand::<C>(&mut case_child, ret, self.out.clone());
        self.cases.push((value.into(), case_child));
        self
    }

    pub fn finish(self, scope: &mut Scope) -> ExpandElementTyped<C> {
        let value_var = *self.value.expand;
        scope.register(Branch::Switch(Box::new(Switch {
            value: value_var,
            scope_default: self.default,
            cases: self
                .cases
                .into_iter()
                .map(|it| (*it.0.expand, it.1))
                .collect(),
        })));
        self.out
    }
}

pub fn switch_expand_expr<I: Int, C: CubePrimitive>(
    scope: &mut Scope,
    value: ExpandElementTyped<I>,
    default_block: impl FnOnce(&mut Scope) -> ExpandElementTyped<C>,
) -> SwitchExpandExpr<I, C> {
    let mut default_child = scope.child();
    let default = default_block(&mut default_child);
    let out: ExpandElementTyped<C> = scope.create_local_mut(default.expand.item).into();
    assign::expand::<C>(&mut default_child, default, out.clone());

    SwitchExpandExpr {
        value,
        out,
        default: default_child,
        cases: Vec::new(),
    }
}

pub fn break_expand(scope: &mut Scope) {
    scope.register(Branch::Break);
}

pub fn return_expand(scope: &mut Scope) {
    scope.register(Branch::Return);
}

// Don't make this `FnOnce`, it must be executable multiple times
pub fn loop_expand(scope: &mut Scope, mut block: impl FnMut(&mut Scope)) {
    let mut inside_loop = scope.child();

    block(&mut inside_loop);
    scope.register(Branch::Loop(Box::new(Loop { scope: inside_loop })));
}

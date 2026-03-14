use alloc::{boxed::Box, vec::Vec};
use cubecl_ir::ManagedVariable;
use num_traits::NumCast;

use crate::{ir::Switch, prelude::CubeEnum};
use crate::{
    ir::{Branch, If, IfElse, Loop, RangeLoop, Scope},
    prelude::Assign,
};

use super::{CubeType, Int, NativeExpand, Numeric};

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
    /// Return the comptime length of this iterable, if possible
    fn const_len(&self) -> Option<usize> {
        None
    }
}

pub struct RangeExpand<I: Int> {
    pub start: NativeExpand<I>,
    pub end: NativeExpand<I>,
    pub inclusive: bool,
}

impl<I: Int> RangeExpand<I> {
    pub fn new(start: NativeExpand<I>, end: NativeExpand<I>, inclusive: bool) -> Self {
        RangeExpand {
            start,
            end,
            inclusive,
        }
    }

    pub fn __expand_step_by_method(self, n: impl Into<NativeExpand<I>>) -> SteppedRangeExpand<I> {
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
        let index_ty = I::as_type(scope);
        let i = child.create_local_restricted(index_ty);

        body(&mut child, i.clone().into());

        let mut start = *self.start.expand;
        let mut end = *self.end.expand;

        // Normalize usize constants. Gotta fix this properly at some point.
        start.ty = I::as_type(scope);
        end.ty = I::as_type(scope);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start,
            end,
            step: None,
            scope: child,
            inclusive: self.inclusive,
        })));
    }

    fn const_len(&self) -> Option<usize> {
        let start = self.start.expand.as_const()?.as_i64();
        let end = self.end.expand.as_const()?.as_i64();
        Some(start.abs_diff(end) as usize)
    }
}

pub struct SteppedRangeExpand<I: Int> {
    start: NativeExpand<I>,
    end: NativeExpand<I>,
    step: NativeExpand<I>,
    inclusive: bool,
}

impl<I: Int + Into<ManagedVariable>> Iterable<I> for SteppedRangeExpand<I> {
    fn expand(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <I as CubeType>::ExpandType),
    ) {
        let mut child = scope.child();
        let index_ty = I::as_type(scope);
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
            .as_i128();
        let end = self
            .end
            .expand
            .as_const()
            .expect("Only constant end can be unrolled.")
            .as_i128();
        let step = self
            .step
            .expand
            .as_const()
            .expect("Only constant step can be unrolled.")
            .as_i128();

        match (self.inclusive, step.is_negative()) {
            (true, true) => {
                for i in (end..=start).rev().step_by(step.unsigned_abs() as usize) {
                    let var = I::from_int_128(i);
                    body(scope, var.into())
                }
            }
            (true, false) => {
                for i in (start..=end).step_by(step.unsigned_abs() as usize) {
                    let var = I::from_int_128(i);
                    body(scope, var.into())
                }
            }
            (false, true) => {
                for i in (end..start).rev().step_by(step.unsigned_abs() as usize) {
                    let var = I::from_int_128(i);
                    body(scope, var.into())
                }
            }
            (false, false) => {
                for i in (start..end).step_by(step.unsigned_abs() as usize) {
                    let var = I::from_int_128(i);
                    body(scope, var.into())
                }
            }
        }
    }

    fn const_len(&self) -> Option<usize> {
        let start = self.start.constant()?.as_i128();
        let end = self.end.constant()?.as_i128();
        let step = self.step.constant()?.as_i128().unsigned_abs();
        Some((start.abs_diff(end) / step) as usize)
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

    use crate::prelude::{Int, NativeExpand};

    use super::RangeExpand;

    pub fn expand<I: Int>(
        _scope: &mut Scope,
        start: NativeExpand<I>,
        end: NativeExpand<I>,
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
pub fn range_stepped<I: Int>(start: I, end: I, step: I) -> Box<dyn Iterator<Item = I>> {
    let start = start.to_i128().unwrap();
    let end = end.to_i128().unwrap();
    let step = step.to_i128().unwrap();

    if step < 0 {
        Box::new(
            (end..start)
                .rev()
                .step_by(step.unsigned_abs() as usize)
                .map(<I as NumCast>::from)
                .map(Option::unwrap),
        )
    } else {
        Box::new(
            (start..end)
                .step_by(step.unsigned_abs() as usize)
                .map(<I as NumCast>::from)
                .map(Option::unwrap),
        )
    }
}

pub mod range_stepped {
    use cubecl_ir::Scope;

    use crate::prelude::{Int, NativeExpand};

    use super::SteppedRangeExpand;

    pub fn expand<I: Int>(
        _scope: &mut Scope,
        start: NativeExpand<I>,
        end: NativeExpand<I>,
        step: NativeExpand<I>,
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
    body: impl FnMut(&mut Scope, NativeExpand<I>),
) {
    if unroll || range.const_len() == Some(1) {
        range.expand_unroll(scope, body);
    } else {
        range.expand(scope, body);
    }
}

pub fn if_expand(scope: &mut Scope, condition: NativeExpand<bool>, block: impl FnOnce(&mut Scope)) {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
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
                cond: *condition.expand,
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
        runtime_cond: NativeExpand<bool>,
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
                    cond: *runtime_cond.expand,
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
    condition: NativeExpand<bool>,
    then_block: impl FnOnce(&mut Scope),
) -> IfElseExpand {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
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
                runtime_cond: condition,
                then_child,
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum IfElseExprExpand<C: Assign> {
    ComptimeThen(C),
    ComptimeElse,
    Runtime {
        runtime_cond: NativeExpand<bool>,
        out: C,
        then_child: Scope,
    },
}

impl<C: Assign> IfElseExprExpand<C> {
    pub fn or_else(self, scope: &mut Scope, else_block: impl FnOnce(&mut Scope) -> C) -> C {
        match self {
            Self::Runtime {
                runtime_cond,
                mut out,
                then_child,
            } => {
                let mut else_child = scope.child();
                let ret = else_block(&mut else_child);
                out.expand_assign(&mut else_child, ret);

                scope.register(Branch::IfElse(Box::new(IfElse {
                    cond: *runtime_cond.expand,
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

pub fn if_else_expr_expand<C: Assign>(
    scope: &mut Scope,
    condition: NativeExpand<bool>,
    then_block: impl FnOnce(&mut Scope) -> C,
) -> IfElseExprExpand<C> {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            let ret = then_block(scope);
            IfElseExprExpand::ComptimeThen(ret)
        }
        Some(false) => IfElseExprExpand::ComptimeElse,
        None => {
            let mut then_child = scope.child();
            let ret = then_block(&mut then_child);
            let mut out = ret.init_mut(scope);
            out.expand_assign(&mut then_child, ret);

            IfElseExprExpand::Runtime {
                runtime_cond: condition,
                out,
                then_child,
            }
        }
    }
}

pub struct SwitchExpand<I: Int> {
    value: NativeExpand<I>,
    default: Scope,
    cases: Vec<(NativeExpand<I>, Scope)>,
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
    value: NativeExpand<I>,
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

pub struct SwitchExpandExpr<I: Int, C: Assign> {
    value: NativeExpand<I>,
    out: C,
    default: Scope,
    cases: Vec<(NativeExpand<I>, Scope)>,
}

impl<I: Int, C: Assign> SwitchExpandExpr<I, C> {
    pub fn case(
        mut self,
        scope: &mut Scope,
        value: impl Int,
        block: impl FnOnce(&mut Scope) -> C,
    ) -> Self {
        let value = I::from(value).unwrap();
        let mut case_child = scope.child();
        let ret = block(&mut case_child);
        self.out.expand_assign(&mut case_child, ret);
        self.cases.push((value.into(), case_child));
        self
    }

    pub fn finish(self, scope: &mut Scope) -> C {
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

pub fn switch_expand_expr<I: Int, C: Assign>(
    scope: &mut Scope,
    value: NativeExpand<I>,
    default_block: impl FnOnce(&mut Scope) -> C,
) -> SwitchExpandExpr<I, C> {
    let mut default_child = scope.child();
    let default = default_block(&mut default_child);
    let mut out = default.init_mut(scope);
    out.expand_assign(&mut default_child, default);

    SwitchExpandExpr {
        value,
        out,
        default: default_child,
        cases: Vec::new(),
    }
}

#[allow(clippy::large_enum_variant)]
pub enum MatchExpand<T: CubeEnum> {
    ComptimeVariant {
        variant: i32,
        runtime_value: T::RuntimeValue,
        matched: bool,
    },
    RuntimeVariant {
        variant: NativeExpand<i32>,
        cases: Vec<(NativeExpand<i32>, Scope)>,
        runtime_value: T::RuntimeValue,
        default: Option<Scope>,
    },
}

impl<T: CubeEnum> MatchExpand<T> {
    pub fn case(
        mut self,
        scope: &mut Scope,
        value: i32,
        block: impl FnOnce(&mut Scope, T::RuntimeValue),
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                cases,
                runtime_value,
                ..
            } => {
                let mut case_child = scope.child();
                block(&mut case_child, runtime_value.clone());
                cases.push((value.into(), case_child));
            }
            Self::ComptimeVariant {
                variant,
                runtime_value,
                matched,
            } => {
                if value == *variant {
                    block(scope, runtime_value.clone());
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn default(
        mut self,
        scope: &mut Scope,
        block: impl FnOnce(&mut Scope, T::RuntimeValue),
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                runtime_value,
                default,
                ..
            } => {
                let mut case_child = scope.child();
                block(&mut case_child, runtime_value.clone());
                *default = Some(case_child);
            }
            Self::ComptimeVariant {
                runtime_value,
                matched,
                ..
            } => {
                if !*matched {
                    block(scope, runtime_value.clone());
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn finish(self, scope: &mut Scope) {
        match self {
            MatchExpand::ComptimeVariant { .. } => {}
            MatchExpand::RuntimeVariant {
                variant,
                cases,
                default,
                ..
            } => {
                let variant_var = *variant.expand;
                let scope_default = default.unwrap_or_else(|| {
                    let mut scope_default = scope.child();
                    unreachable_unchecked::expand(&mut scope_default);
                    scope_default
                });

                scope.register(Branch::Switch(Box::new(Switch {
                    value: variant_var,
                    scope_default,
                    cases: cases.into_iter().map(|it| (*it.0.expand, it.1)).collect(),
                })));
            }
        }
    }
}

pub fn match_expand<T: CubeEnum>(
    scope: &mut Scope,
    value: T,
    discriminant0: i32,
    arm0: impl FnOnce(&mut Scope, T::RuntimeValue),
) -> MatchExpand<T> {
    let discriminant = value.discriminant();
    match discriminant.constant() {
        Some(const_variant) if const_variant.as_i32() == discriminant0 => {
            let runtime_value = value.runtime_value();
            arm0(scope, runtime_value.clone());
            MatchExpand::ComptimeVariant {
                variant: const_variant.as_i32(),
                runtime_value,
                matched: true,
            }
        }
        Some(const_variant) => MatchExpand::ComptimeVariant {
            variant: const_variant.as_i32(),
            runtime_value: value.runtime_value(),
            matched: false,
        },
        None => {
            let runtime_value = value.runtime_value();
            let mut case_child = scope.child();
            arm0(&mut case_child, runtime_value.clone());

            MatchExpand::RuntimeVariant {
                variant: discriminant,
                cases: alloc::vec![(discriminant0.into(), case_child)],
                runtime_value,
                default: None,
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum MatchExpandExpr<T: CubeEnum, C: Assign> {
    ComptimeVariant {
        variant: i32,
        runtime_value: T::RuntimeValue,
        out: Option<C>,
        matched: bool,
    },
    RuntimeVariant {
        variant: NativeExpand<i32>,
        out: C,
        cases: Vec<(NativeExpand<i32>, Scope)>,
        runtime_value: T::RuntimeValue,
        default: Option<Scope>,
    },
}

impl<T: CubeEnum, C: Assign> MatchExpandExpr<T, C> {
    pub fn case(
        mut self,
        scope: &mut Scope,
        value: i32,
        block: impl FnOnce(&mut Scope, T::RuntimeValue) -> C,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                cases,
                out,
                runtime_value,
                ..
            } => {
                let mut case_child = scope.child();
                let ret_val = block(&mut case_child, runtime_value.clone());
                out.expand_assign(&mut case_child, ret_val);
                cases.push((value.into(), case_child));
            }
            Self::ComptimeVariant {
                variant,
                runtime_value,
                out,
                matched,
            } => {
                if value == *variant {
                    *out = Some(block(scope, runtime_value.clone()));
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn default(
        mut self,
        scope: &mut Scope,
        block: impl FnOnce(&mut Scope, T::RuntimeValue) -> C,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                runtime_value,
                out,
                default,
                ..
            } => {
                let mut case_child = scope.child();
                let ret_val = block(&mut case_child, runtime_value.clone());
                out.expand_assign(&mut case_child, ret_val);
                *default = Some(case_child);
            }
            Self::ComptimeVariant {
                runtime_value,
                out,
                matched,
                ..
            } => {
                if !*matched {
                    *out = Some(block(scope, runtime_value.clone()));
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn finish(self, scope: &mut Scope) -> C {
        match self {
            MatchExpandExpr::ComptimeVariant { out, .. } => {
                out.expect("At least one variant should be matched")
            }
            MatchExpandExpr::RuntimeVariant {
                variant,
                cases,
                out,
                default,
                ..
            } => {
                let variant_var = *variant.expand;
                let scope_default = default.unwrap_or_else(|| {
                    let mut scope_default = scope.child();
                    unreachable_unchecked::expand(&mut scope_default);
                    scope_default
                });
                scope.register(Branch::Switch(Box::new(Switch {
                    value: variant_var,
                    scope_default,
                    cases: cases.into_iter().map(|it| (*it.0.expand, it.1)).collect(),
                })));
                out
            }
        }
    }
}

pub fn match_expand_expr<T: CubeEnum, C: Assign>(
    scope: &mut Scope,
    value: T,
    discriminant0: i32,
    arm0: impl FnOnce(&mut Scope, T::RuntimeValue) -> C,
) -> MatchExpandExpr<T, C> {
    let discriminant = value.discriminant();
    match discriminant.constant() {
        Some(const_variant) if const_variant.as_i32() == discriminant0 => {
            let runtime_value = value.runtime_value();
            let out = arm0(scope, runtime_value.clone());
            MatchExpandExpr::ComptimeVariant {
                variant: const_variant.as_i32(),
                out: Some(out),
                runtime_value,
                matched: true,
            }
        }
        Some(const_variant) => MatchExpandExpr::ComptimeVariant {
            variant: const_variant.as_i32(),
            out: None,
            runtime_value: value.runtime_value(),
            matched: false,
        },
        None => {
            let runtime_value = value.runtime_value();
            let mut case_child = scope.child();
            let ret_val = arm0(&mut case_child, runtime_value.clone());

            let mut out = ret_val.init_mut(scope);
            out.expand_assign(&mut case_child, ret_val);

            MatchExpandExpr::RuntimeVariant {
                variant: discriminant,
                out,
                cases: alloc::vec![(discriminant0.into(), case_child)],
                runtime_value,
                default: None,
            }
        }
    }
}

pub fn break_expand(scope: &mut Scope) {
    scope.register(Branch::Break);
}

pub fn return_expand(scope: &mut Scope) {
    scope.register(Branch::Return);
}

pub mod unreachable_unchecked {
    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(Branch::Unreachable);
    }
}

// Don't make this `FnOnce`, it must be executable multiple times
pub fn loop_expand(scope: &mut Scope, mut block: impl FnMut(&mut Scope)) {
    let mut inside_loop = scope.child();

    block(&mut inside_loop);
    scope.register(Branch::Loop(Box::new(Loop { scope: inside_loop })));
}

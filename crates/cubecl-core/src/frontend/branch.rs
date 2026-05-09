use alloc::{boxed::Box, vec::Vec};

use crate::{
    ir::Switch,
    prelude::{CubeEnum, ExpandTypeClone},
};
use crate::{
    ir::{Branch, If, IfElse, Loop, Scope},
    prelude::Assign,
};

use super::{Int, NativeExpand};

/// Something that can be iterated on by a for loop. Currently only includes `Range`, `StepBy` and
/// `Sequence`.
pub trait Iterable: Sized {
    type Item;

    /// Expand a runtime loop without unrolling
    ///
    /// # Arguments
    /// # Arguments
    /// * `scope` - the expansion scope
    /// * `body` - the loop body to be executed repeatedly
    fn expand(self, scope: &Scope, body: impl FnMut(&Scope, Self::Item));
    /// Expand an unrolled loop. The body should be invoced `n` times, where `n` is the number of
    /// iterations.
    ///
    /// # Arguments
    /// * `scope` - the expansion scope
    /// * `body` - the loop body to be executed repeatedly
    fn expand_unroll(self, scope: &Scope, body: impl FnMut(&Scope, Self::Item));
    /// Return the comptime length of this iterable, if possible
    fn const_len(&self) -> Option<usize> {
        None
    }
}

pub fn for_expand<I: Iterable>(
    scope: &Scope,
    range: I,
    unroll: bool,
    body: impl FnMut(&Scope, I::Item),
) {
    if unroll || range.const_len() == Some(1) {
        range.expand_unroll(scope, body);
    } else {
        range.expand(scope, body);
    }
}

pub fn if_expand(scope: &Scope, condition: NativeExpand<bool>, block: impl FnOnce(&Scope)) {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(cond) => {
            if cond {
                block(scope);
            }
        }
        None => {
            let child = scope.child();

            block(&child);

            scope.register(Branch::If(Box::new(If {
                cond: condition.expand,
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
    pub fn or_else(self, scope: &Scope, else_block: impl FnOnce(&Scope)) {
        match self {
            Self::Runtime {
                runtime_cond,
                then_child,
            } => {
                let else_child = scope.child();
                else_block(&else_child);

                scope.register(Branch::IfElse(Box::new(IfElse {
                    cond: runtime_cond.expand,
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
    scope: &Scope,
    condition: NativeExpand<bool>,
    then_block: impl FnOnce(&Scope),
) -> IfElseExpand {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            then_block(scope);
            IfElseExpand::ComptimeThen
        }
        Some(false) => IfElseExpand::ComptimeElse,
        None => {
            let then_child = scope.child();
            then_block(&then_child);

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
    pub fn or_else(self, scope: &Scope, else_block: impl FnOnce(&Scope) -> C) -> C {
        match self {
            Self::Runtime {
                runtime_cond,
                mut out,
                then_child,
            } => {
                let else_child = scope.child();
                let ret = else_block(&else_child);
                out.__expand_assign_method(&else_child, ret);

                scope.register(Branch::IfElse(Box::new(IfElse {
                    cond: runtime_cond.expand,
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
    scope: &Scope,
    condition: NativeExpand<bool>,
    then_block: impl FnOnce(&Scope) -> C,
) -> IfElseExprExpand<C> {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            let ret = then_block(scope);
            IfElseExprExpand::ComptimeThen(ret)
        }
        Some(false) => IfElseExprExpand::ComptimeElse,
        None => {
            let then_child = scope.child();
            let ret = then_block(&then_child);
            let mut out = ret.init_mut(scope);
            out.__expand_assign_method(&then_child, ret);

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
    pub fn case(mut self, scope: &Scope, value: impl Int, block: impl FnOnce(&Scope)) -> Self {
        let value = I::from(value).unwrap();
        let case_child = scope.child();
        block(&case_child);
        self.cases.push((value.into(), case_child));
        self
    }

    pub fn finish(self, scope: &Scope) {
        let value_var = self.value.expand;
        scope.register(Branch::Switch(Box::new(Switch {
            value: value_var,
            scope_default: self.default,
            cases: self
                .cases
                .into_iter()
                .map(|it| (it.0.expand, it.1))
                .collect(),
        })));
    }
}

pub fn switch_expand<I: Int>(
    scope: &Scope,
    value: NativeExpand<I>,
    default_block: impl FnOnce(&Scope),
) -> SwitchExpand<I> {
    let default_child = scope.child();
    default_block(&default_child);

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
    pub fn case(mut self, scope: &Scope, value: impl Int, block: impl FnOnce(&Scope) -> C) -> Self {
        let value = I::from(value).unwrap();
        let case_child = scope.child();
        let ret = block(&case_child);
        self.out.__expand_assign_method(&case_child, ret);
        self.cases.push((value.into(), case_child));
        self
    }

    pub fn finish(self, scope: &Scope) -> C {
        let value_var = self.value.expand;
        scope.register(Branch::Switch(Box::new(Switch {
            value: value_var,
            scope_default: self.default,
            cases: self
                .cases
                .into_iter()
                .map(|it| (it.0.expand, it.1))
                .collect(),
        })));
        self.out
    }
}

pub fn switch_expand_expr<I: Int, C: Assign>(
    scope: &Scope,
    value: NativeExpand<I>,
    default_block: impl FnOnce(&Scope) -> C,
) -> SwitchExpandExpr<I, C> {
    let default_child = scope.child();
    let default = default_block(&default_child);
    let mut out = default.init_mut(scope);
    out.__expand_assign_method(&default_child, default);

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
        scope: &Scope,
        value: i32,
        block: impl FnOnce(&Scope, T::RuntimeValue),
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                cases,
                runtime_value,
                ..
            } => {
                let case_child = scope.child();
                block(&case_child, (*runtime_value).clone_unchecked());
                cases.push((value.into(), case_child));
            }
            Self::ComptimeVariant {
                variant,
                runtime_value,
                matched,
            } => {
                if value == *variant {
                    block(scope, (*runtime_value).clone_unchecked());
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn default(mut self, scope: &Scope, block: impl FnOnce(&Scope, T::RuntimeValue)) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                runtime_value,
                default,
                ..
            } => {
                let case_child = scope.child();
                block(&case_child, (*runtime_value).clone_unchecked());
                *default = Some(case_child);
            }
            Self::ComptimeVariant {
                runtime_value,
                matched,
                ..
            } => {
                if !*matched {
                    block(scope, (*runtime_value).clone_unchecked());
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn finish(self, scope: &Scope) {
        match self {
            MatchExpand::ComptimeVariant { .. } => {}
            MatchExpand::RuntimeVariant {
                variant,
                cases,
                default,
                ..
            } => {
                let variant_var = variant.expand;
                let scope_default = default.unwrap_or_else(|| {
                    let scope_default = scope.child();
                    unreachable_unchecked::expand(&scope_default);
                    scope_default
                });

                scope.register(Branch::Switch(Box::new(Switch {
                    value: variant_var,
                    scope_default,
                    cases: cases.into_iter().map(|it| (it.0.expand, it.1)).collect(),
                })));
            }
        }
    }
}

pub fn match_expand<T: CubeEnum>(
    scope: &Scope,
    value: T,
    discriminant0: i32,
    arm0: impl FnOnce(&Scope, T::RuntimeValue),
) -> MatchExpand<T> {
    let discriminant = value.discriminant();
    match discriminant.constant() {
        Some(const_variant) if const_variant.as_i32() == discriminant0 => {
            let runtime_value = value.runtime_value();
            arm0(scope, runtime_value.clone_unchecked());
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
            let case_child = scope.child();
            arm0(&case_child, runtime_value.clone_unchecked());

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
        scope: &Scope,
        value: i32,
        block: impl FnOnce(&Scope, T::RuntimeValue) -> C,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                cases,
                out,
                runtime_value,
                ..
            } => {
                let case_child = scope.child();
                let ret_val = block(&case_child, (*runtime_value).clone_unchecked());
                out.__expand_assign_method(&case_child, ret_val);
                cases.push((value.into(), case_child));
            }
            Self::ComptimeVariant {
                variant,
                runtime_value,
                out,
                matched,
            } => {
                if value == *variant {
                    *out = Some(block(scope, (*runtime_value).clone_unchecked()));
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn default(
        mut self,
        scope: &Scope,
        block: impl FnOnce(&Scope, T::RuntimeValue) -> C,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                runtime_value,
                out,
                default,
                ..
            } => {
                let case_child = scope.child();
                let ret_val = block(&case_child, (*runtime_value).clone_unchecked());
                out.__expand_assign_method(&case_child, ret_val);
                *default = Some(case_child);
            }
            Self::ComptimeVariant {
                runtime_value,
                out,
                matched,
                ..
            } => {
                if !*matched {
                    *out = Some(block(scope, (*runtime_value).clone_unchecked()));
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn finish(self, scope: &Scope) -> C {
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
                let variant_var = variant.expand;
                let scope_default = default.unwrap_or_else(|| {
                    let scope_default = scope.child();
                    unreachable_unchecked::expand(&scope_default);
                    scope_default
                });
                scope.register(Branch::Switch(Box::new(Switch {
                    value: variant_var,
                    scope_default,
                    cases: cases.into_iter().map(|it| (it.0.expand, it.1)).collect(),
                })));
                out
            }
        }
    }
}

pub fn match_expand_expr<T: CubeEnum, C: Assign>(
    scope: &Scope,
    value: T,
    discriminant0: i32,
    arm0: impl FnOnce(&Scope, T::RuntimeValue) -> C,
) -> MatchExpandExpr<T, C> {
    let discriminant = value.discriminant();
    match discriminant.constant() {
        Some(const_variant) if const_variant.as_i32() == discriminant0 => {
            let runtime_value = value.runtime_value();
            let out = arm0(scope, runtime_value.clone_unchecked());
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
            let case_child = scope.child();
            let ret_val = arm0(&case_child, runtime_value.clone_unchecked());

            let mut out = ret_val.init_mut(scope);
            out.__expand_assign_method(&case_child, ret_val);

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

pub fn break_expand(scope: &Scope) {
    scope.register(Branch::Break);
}

pub fn return_expand(scope: &Scope) {
    scope.register(Branch::Return);
}

pub mod unreachable_unchecked {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(Branch::Unreachable);
    }
}

// Don't make this `FnOnce`, it must be executable multiple times
pub fn loop_expand(scope: &Scope, mut block: impl FnMut(&Scope)) {
    let inside_loop = scope.child();

    block(&inside_loop);
    scope.register(Branch::Loop(Box::new(Loop { scope: inside_loop })));
}

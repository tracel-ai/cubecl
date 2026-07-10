use alloc::vec::Vec;
use cubecl_ir::{
    OpInserter,
    dialect::branch::{BreakOp, IfOp, LoopOp, ReturnOp, SwitchOp, UnreachableOp},
    pliron::{irbuild::inserter::Inserter, r#type::TypedHandle},
};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        types::{IntegerType, Signedness},
    },
    utils::apint::{APInt, bw},
};

use crate::{
    frontend::{ReadValue, RuntimeAssign},
    prelude::{CubeEnum, ExpandTypeClone},
};
use crate::{ir::Scope, prelude::Assign};

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
            let cond = condition.read_value(scope);
            let if_op = IfOp::new(scope.ctx_mut(), cond);

            let then_block = if_op.then_block(scope.ctx());
            let then_child = scope.child(OpInserter::new_at_block_end(then_block));
            block(&then_child);
            then_child.terminate_yield();

            let else_block = if_op.else_block(scope.ctx());
            let else_child = scope.child(OpInserter::new_at_block_end(else_block));
            else_child.terminate_yield();

            scope.register(&if_op);
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum IfElseExpand {
    ComptimeThen,
    ComptimeElse,
    Runtime {
        runtime_cond: NativeExpand<bool>,
        if_op: IfOp,
    },
}

impl IfElseExpand {
    pub fn or_else(self, scope: &Scope, else_block: impl FnOnce(&Scope)) {
        match self {
            Self::Runtime { if_op, .. } => {
                let else_body = if_op.else_block(scope.ctx());
                let else_child = scope.child(OpInserter::new_at_block_end(else_body));
                else_block(&else_child);
                else_child.terminate_yield();

                scope.register(&if_op);
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
            let cond = condition.read_value(scope);
            let if_op = IfOp::new(scope.ctx_mut(), cond);
            let if_block = if_op.then_block(scope.ctx());
            let then_child = scope.child(OpInserter::new_at_block_end(if_block));
            then_block(&then_child);
            then_child.terminate_yield();

            IfElseExpand::Runtime {
                runtime_cond: condition,
                if_op,
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
        if_op: IfOp,
    },
}

impl<C: Assign> IfElseExprExpand<C> {
    pub fn or_else<R: RuntimeAssign<Expand = C>>(
        self,
        scope: &Scope,
        else_block: impl FnOnce(&Scope) -> R,
    ) -> C {
        match self {
            Self::Runtime { mut out, if_op, .. } => {
                let else_body = if_op.else_block(scope.ctx());
                let else_child = scope.child(OpInserter::new_at_block_end(else_body));
                let ret = else_block(&else_child);
                out.__expand_assign_method(&else_child, ret.into_expand(scope));
                else_child.terminate_yield();

                scope.register(&if_op);
                out
            }
            Self::ComptimeElse => else_block(scope).into_expand(scope),
            Self::ComptimeThen(ret) => ret,
        }
    }
}

pub fn if_else_expr_expand<C: RuntimeAssign>(
    scope: &Scope,
    condition: NativeExpand<bool>,
    then_block: impl FnOnce(&Scope) -> C,
) -> IfElseExprExpand<C::Expand> {
    let comptime_cond = condition.expand.as_const().map(|it| it.as_bool());
    match comptime_cond {
        Some(true) => {
            let ret = then_block(scope);
            IfElseExprExpand::ComptimeThen(ret.into_expand(scope))
        }
        Some(false) => IfElseExprExpand::ComptimeElse,
        None => {
            let cond = condition.read_value(scope);
            let if_op = IfOp::new(scope.ctx_mut(), cond);
            let then_body = if_op.then_block(scope.ctx());
            let then_child = scope.child(OpInserter::new_at_block_end(then_body));
            let ret = then_block(&then_child);
            let mut out = ret.init_mut(scope);
            out.__expand_assign_method(&then_child, ret.into_expand(scope));
            then_child.terminate_yield();

            IfElseExprExpand::Runtime {
                runtime_cond: condition,
                out,
                if_op,
            }
        }
    }
}

pub struct SwitchExpand<I: Int> {
    switch_op: SwitchOp,
    cases: Vec<I>,
}

impl<I: Int> SwitchExpand<I> {
    pub fn case(mut self, scope: &Scope, value: impl Int, block: impl FnOnce(&Scope)) -> Self {
        let value = I::from(value).unwrap();
        self.cases.push(value);
        let body = self.switch_op.append_case_block(scope.ctx_mut());
        let case_child = scope.child(OpInserter::new_at_block_end(body));
        block(&case_child);
        case_child.terminate_yield();
        self
    }

    pub fn finish(self, scope: &Scope) {
        let cases = self.cases.into_iter().map(|case| {
            let ty = I::__expand_as_type(scope);
            let ty = TypedHandle::<IntegerType>::from_handle(ty, scope.ctx()).unwrap();
            let width = bw(ty.deref(scope.ctx()).width() as usize);
            let val = APInt::from_i128(case.to_i128().unwrap(), width);
            IntegerAttr::new(ty, val).into()
        });
        self.switch_op.set_attr_cases(scope.ctx(), cases);
        scope.register(&self.switch_op);
    }
}

pub fn switch_expand<I: Int>(
    scope: &Scope,
    value: NativeExpand<I>,
    default_block: impl FnOnce(&Scope),
) -> SwitchExpand<I> {
    let value = value.read_value(scope);
    let switch_op = SwitchOp::new(scope.ctx_mut(), value);

    let default_body = switch_op.default_block(scope.ctx());
    let default_child = scope.child(OpInserter::new_at_block_end(default_body));
    default_block(&default_child);
    default_child.terminate_yield();

    SwitchExpand {
        switch_op,
        cases: Vec::new(),
    }
}

pub struct SwitchExpandExpr<I: Int, C: Assign> {
    switch_op: SwitchOp,
    cases: Vec<I>,
    out: C,
}

impl<I: Int, C: Assign> SwitchExpandExpr<I, C> {
    pub fn case<T: RuntimeAssign<Expand = C>>(
        mut self,
        scope: &Scope,
        value: impl Int,
        block: impl FnOnce(&Scope) -> T,
    ) -> Self {
        let value = I::from(value).unwrap();
        self.cases.push(value);
        let body = self.switch_op.append_case_block(scope.ctx_mut());
        let case_child = scope.child(OpInserter::new_at_block_end(body));
        let ret = block(&case_child);
        self.out
            .__expand_assign_method(&case_child, ret.into_expand(scope));
        case_child.terminate_yield();
        self
    }

    pub fn finish(self, scope: &Scope) -> C {
        let cases = self.cases.into_iter().map(|case| {
            let ty = I::__expand_as_type(scope);
            let ty = TypedHandle::<IntegerType>::from_handle(ty, scope.ctx()).unwrap();
            let width = bw(ty.deref(scope.ctx()).width() as usize);
            let val = APInt::from_i128(case.to_i128().unwrap(), width);
            IntegerAttr::new(ty, val).into()
        });
        self.switch_op.set_attr_cases(scope.ctx(), cases);
        scope.register(&self.switch_op);
        self.out
    }
}

pub fn switch_expand_expr<I: Int, C: RuntimeAssign>(
    scope: &Scope,
    value: NativeExpand<I>,
    default_block: impl FnOnce(&Scope) -> C,
) -> SwitchExpandExpr<I, C::Expand> {
    let value = value.read_value(scope);
    let switch_op = SwitchOp::new(scope.ctx_mut(), value);

    let default_body = switch_op.default_block(scope.ctx());
    let default_child = scope.child(OpInserter::new_at_block_end(default_body));
    let default = default_block(&default_child);
    let mut out = default.init_mut(scope);
    out.__expand_assign_method(&default_child, default.into_expand(scope));
    default_child.terminate_yield();

    SwitchExpandExpr {
        switch_op,
        cases: Vec::new(),
        out,
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
        switch_op: SwitchOp,
        cases: Vec<i32>,
        has_default: bool,
        runtime_value: T::RuntimeValue,
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
                switch_op,
                cases,
                runtime_value,
                ..
            } => {
                cases.push(value);
                let body = switch_op.append_case_block(scope.ctx_mut());
                let case_child = scope.child(OpInserter::new_at_block_end(body));
                block(&case_child, (*runtime_value).clone_unchecked());
                case_child.terminate_yield();
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
                switch_op,
                runtime_value,
                has_default,
                ..
            } => {
                let body = switch_op.default_block(scope.ctx());
                let case_child = scope.child(OpInserter::new_at_block_end(body));
                block(&case_child, (*runtime_value).clone_unchecked());
                case_child.terminate_yield();
                *has_default = true;
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
                switch_op,
                cases,
                has_default,
                ..
            } => {
                if !has_default {
                    let default_body = switch_op.default_block(scope.ctx());
                    let mut inserter = OpInserter::new_at_block_end(default_body);
                    let unreachable = UnreachableOp::new(scope.ctx_mut());
                    inserter.append_op(scope.ctx(), &unreachable);
                }

                let cases = cases.into_iter().map(|case| {
                    let ty = IntegerType::get(scope.ctx(), 32, Signedness::Unsigned);
                    IntegerAttr::new(ty, APInt::from_i32(case, bw(32))).into()
                });
                switch_op.set_attr_cases(scope.ctx(), cases);
                scope.register(&switch_op);
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
            let discriminant = discriminant.read_value(scope);
            let runtime_value = value.runtime_value();

            let switch_op = SwitchOp::new(scope.ctx_mut(), discriminant);
            let body = switch_op.append_case_block(scope.ctx_mut());
            let case_child = scope.child(OpInserter::new_at_block_end(body));
            arm0(&case_child, runtime_value.clone_unchecked());
            case_child.terminate_yield();

            MatchExpand::RuntimeVariant {
                switch_op,
                cases: alloc::vec![discriminant0],
                has_default: false,
                runtime_value,
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
        switch_op: SwitchOp,
        cases: Vec<i32>,
        has_default: bool,
        out: C,
        runtime_value: T::RuntimeValue,
    },
}

impl<T: CubeEnum, C: Assign> MatchExpandExpr<T, C> {
    pub fn case<R: RuntimeAssign<Expand = C>>(
        mut self,
        scope: &Scope,
        value: i32,
        block: impl FnOnce(&Scope, T::RuntimeValue) -> R,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                switch_op,
                cases,
                out,
                runtime_value,
                ..
            } => {
                cases.push(value);
                let body = switch_op.append_case_block(scope.ctx_mut());
                let case_child = scope.child(OpInserter::new_at_block_end(body));
                let ret_val = block(&case_child, (*runtime_value).clone_unchecked());
                out.__expand_assign_method(&case_child, ret_val.into_expand(scope));
                case_child.terminate_yield();
            }
            Self::ComptimeVariant {
                variant,
                runtime_value,
                out,
                matched,
            } => {
                if value == *variant {
                    *out =
                        Some(block(scope, (*runtime_value).clone_unchecked()).into_expand(scope));
                    *matched = true;
                }
            }
        }
        self
    }

    pub fn default<R: RuntimeAssign<Expand = C>>(
        mut self,
        scope: &Scope,
        block: impl FnOnce(&Scope, T::RuntimeValue) -> R,
    ) -> Self {
        match &mut self {
            Self::RuntimeVariant {
                switch_op,
                runtime_value,
                out,
                has_default,
                ..
            } => {
                let body = switch_op.default_block(scope.ctx());
                let case_child = scope.child(OpInserter::new_at_block_end(body));
                let ret_val = block(&case_child, (*runtime_value).clone_unchecked());
                out.__expand_assign_method(&case_child, ret_val.into_expand(scope));
                case_child.terminate_yield();
                *has_default = true;
            }
            Self::ComptimeVariant {
                runtime_value,
                out,
                matched,
                ..
            } => {
                if !*matched {
                    *out =
                        Some(block(scope, (*runtime_value).clone_unchecked()).into_expand(scope));
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
                switch_op,
                cases,
                has_default,
                out,
                ..
            } => {
                if !has_default {
                    let default_body = switch_op.default_block(scope.ctx());
                    let mut inserter = OpInserter::new_at_block_end(default_body);
                    let unreachable = UnreachableOp::new(scope.ctx_mut());
                    inserter.append_op(scope.ctx(), &unreachable);
                }

                let cases = cases.into_iter().map(|case| {
                    let ty = IntegerType::get(scope.ctx(), 32, Signedness::Unsigned);
                    IntegerAttr::new(ty, APInt::from_i32(case, bw(32))).into()
                });
                switch_op.set_attr_cases(scope.ctx(), cases);
                scope.register(&switch_op);

                out
            }
        }
    }
}

pub fn match_expand_expr<T: CubeEnum, C: RuntimeAssign>(
    scope: &Scope,
    value: T,
    discriminant0: i32,
    arm0: impl FnOnce(&Scope, T::RuntimeValue) -> C,
) -> MatchExpandExpr<T, C::Expand> {
    let discriminant = value.discriminant();
    match discriminant.constant() {
        Some(const_variant) if const_variant.as_i32() == discriminant0 => {
            let runtime_value = value.runtime_value();
            let out = arm0(scope, runtime_value.clone_unchecked());
            MatchExpandExpr::ComptimeVariant {
                variant: const_variant.as_i32(),
                out: Some(out.into_expand(scope)),
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
            let discriminant = discriminant.read_value(scope);
            let runtime_value = value.runtime_value();

            let switch_op = SwitchOp::new(scope.ctx_mut(), discriminant);
            let body = switch_op.append_case_block(scope.ctx_mut());
            let case_child = scope.child(OpInserter::new_at_block_end(body));
            let ret_val = arm0(&case_child, runtime_value.clone_unchecked());

            let mut out = ret_val.init_mut(scope);
            out.__expand_assign_method(&case_child, ret_val.into_expand(scope));
            case_child.terminate_yield();

            MatchExpandExpr::RuntimeVariant {
                switch_op,
                out,
                cases: alloc::vec![discriminant0],
                runtime_value,
                has_default: false,
            }
        }
    }
}

pub fn break_expand(scope: &Scope) {
    scope.register(&BreakOp::new(scope.ctx_mut()));
}

pub fn return_expand(scope: &Scope) {
    scope.register(&ReturnOp::new(scope.ctx_mut()));
}

pub mod unreachable_unchecked {
    use super::*;

    pub fn expand(scope: &Scope) {
        scope.register(&UnreachableOp::new(scope.ctx_mut()));
    }
}

// Don't make this `FnOnce`, it must be executable multiple times
pub fn loop_expand(scope: &Scope, mut block: impl FnMut(&Scope)) {
    let loop_op = LoopOp::new(scope.ctx_mut());
    let body = loop_op.loop_body(scope.ctx());
    let inside_loop = scope.child(OpInserter::new_at_block_end(body));

    block(&inside_loop);
    inside_loop.terminate_yield();
    scope.register(&loop_op);
}

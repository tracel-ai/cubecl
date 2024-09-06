use std::ops::Deref;

use crate::frontend::{CubeContext, ExpandElement};
use crate::ir::{Branch, Elem, If, IfElse, Item, Loop, RangeLoop, Variable};

use super::ExpandElementTyped;

/// u32 range. Equivalent to:
///
/// ```ignore
/// for i in start..end { ... }
/// ```
pub fn range<S, E>(start: S, end: E, _unroll: bool) -> impl Iterator<Item = u32>
where
    S: Into<u32>,
    E: Into<u32>,
{
    let start: u32 = start.into();
    let end: u32 = end.into();

    start..end
}

/// Stepped range. Equivalent to:
///
/// ```ignore
/// for i in (start..end).step_by(step) { ... }
/// ```
pub fn range_stepped<S, E, Step>(
    start: S,
    end: E,
    step: Step,
    _unroll: bool,
) -> impl Iterator<Item = u32>
where
    S: Into<u32>,
    E: Into<u32>,
    Step: Into<u32>,
{
    let start: u32 = start.into();
    let end: u32 = end.into();
    let step: u32 = step.into();

    (start..end).step_by(step as usize)
}

pub fn range_expand<F, S, E>(context: &mut CubeContext, start: S, end: E, unroll: bool, mut func: F)
where
    F: FnMut(&mut CubeContext, ExpandElementTyped<u32>),
    S: Into<ExpandElementTyped<u32>>,
    E: Into<ExpandElementTyped<u32>>,
{
    let start: ExpandElementTyped<u32> = start.into();
    let end: ExpandElementTyped<u32> = end.into();
    let start = start.expand;
    let end = end.expand;

    if unroll {
        let start = match start.deref() {
            Variable::ConstantScalar(value) => value.as_usize(),
            _ => panic!("Only constant start can be unrolled."),
        };
        let end = match end.deref() {
            Variable::ConstantScalar(value) => value.as_usize(),
            _ => panic!("Only constant end can be unrolled."),
        };

        for i in start..end {
            let var: ExpandElement = i.into();
            func(context, var.into())
        }
    } else {
        let mut child = context.child();
        let index_ty = Item::new(Elem::UInt);
        let i = child.scope.borrow_mut().create_local_undeclared(index_ty);
        let i = ExpandElement::Plain(i);

        func(&mut child, i.clone().into());

        context.register(Branch::RangeLoop(RangeLoop {
            i: *i,
            start: *start,
            end: *end,
            step: None,
            scope: child.into_scope(),
        }));
    }
}

pub fn range_stepped_expand<F, S, E, Step>(
    context: &mut CubeContext,
    start: S,
    end: E,
    step: Step,
    unroll: bool,
    mut func: F,
) where
    F: FnMut(&mut CubeContext, ExpandElementTyped<u32>),
    S: Into<ExpandElementTyped<u32>>,
    E: Into<ExpandElementTyped<u32>>,
    Step: Into<ExpandElementTyped<u32>>,
{
    let start: ExpandElementTyped<u32> = start.into();
    let end: ExpandElementTyped<u32> = end.into();
    let step: ExpandElementTyped<u32> = step.into();
    let start = start.expand;
    let end = end.expand;
    let step = step.expand;

    if unroll {
        let start = match start.deref() {
            Variable::ConstantScalar(value) => value.as_usize(),
            _ => panic!("Only constant start can be unrolled."),
        };
        let end = match end.deref() {
            Variable::ConstantScalar(value) => value.as_usize(),
            _ => panic!("Only constant end can be unrolled."),
        };
        let step: usize = match step.deref() {
            Variable::ConstantScalar(value) => value.as_usize(),
            _ => panic!("Only constant step can be unrolled."),
        };

        for i in (start..end).step_by(step) {
            let var: ExpandElement = i.into();
            func(context, var.into())
        }
    } else {
        let mut child = context.child();
        let index_ty = Item::new(Elem::UInt);
        let i = child.scope.borrow_mut().create_local_undeclared(index_ty);
        let i = ExpandElement::Plain(i);

        func(&mut child, i.clone().into());

        context.register(Branch::RangeLoop(RangeLoop {
            i: *i,
            start: *start,
            end: *end,
            step: Some(*step),
            scope: child.into_scope(),
        }));
    }
}

pub fn if_expand<IF>(
    context: &mut CubeContext,
    comptime_cond: Option<bool>,
    runtime_cond: ExpandElement,
    mut block: IF,
) where
    IF: FnMut(&mut CubeContext),
{
    match comptime_cond {
        Some(cond) => {
            if cond {
                block(context);
            }
        }
        None => {
            let mut child = context.child();

            block(&mut child);

            context.register(Branch::If(If {
                cond: *runtime_cond,
                scope: child.into_scope(),
            }));
        }
    }
}

pub fn if_else_expand<IF, EL>(
    context: &mut CubeContext,
    comptime_cond: Option<bool>,
    runtime_cond: ExpandElement,
    mut then_block: IF,
    mut else_block: EL,
) where
    IF: FnMut(&mut CubeContext),
    EL: FnMut(&mut CubeContext),
{
    match comptime_cond {
        Some(cond) => {
            if cond {
                then_block(context);
            } else {
                else_block(context);
            }
        }
        None => {
            let mut then_child = context.child();
            then_block(&mut then_child);

            let mut else_child = context.child();
            else_block(&mut else_child);

            context.register(Branch::IfElse(IfElse {
                cond: *runtime_cond,
                scope_if: then_child.into_scope(),
                scope_else: else_child.into_scope(),
            }));
        }
    }
}

pub fn break_expand(context: &mut CubeContext) {
    context.register(Branch::Break);
}

pub fn return_expand(context: &mut CubeContext) {
    context.register(Branch::Return);
}

pub fn loop_expand<FB>(context: &mut CubeContext, mut block: FB)
where
    FB: FnMut(&mut CubeContext),
{
    let mut inside_loop = context.child();

    block(&mut inside_loop);
    context.register(Branch::Loop(Loop {
        scope: inside_loop.into_scope(),
    }));
}

pub fn while_loop_expand<FC, FB>(context: &mut CubeContext, mut cond_fn: FC, mut block: FB)
where
    FC: FnMut(&mut CubeContext) -> ExpandElementTyped<bool>,
    FB: FnMut(&mut CubeContext),
{
    let mut inside_loop = context.child();

    let cond: ExpandElement = cond_fn(&mut inside_loop).into();
    if_expand(&mut inside_loop, None, cond, break_expand);

    block(&mut inside_loop);
    context.register(Branch::Loop(Loop {
        scope: inside_loop.into_scope(),
    }));
}

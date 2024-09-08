use num_traits::NumCast;

use crate::frontend::{CubeContext, ExpandElement};
use crate::ir::{Branch, If, IfElse, Item, Loop, RangeLoop};

use super::{CubeType, ExpandElementTyped, Int, Numeric};

/// Something that can be iterated on by a for loop. Currently only includes `Range`, `StepBy` and
/// `Sequence`.
pub trait Iterable<T: CubeType>: Sized {
    fn expand(
        self,
        context: &mut CubeContext,
        func: impl FnMut(&mut CubeContext, <T as CubeType>::ExpandType),
    );
    fn expand_unroll(
        self,
        context: &mut CubeContext,
        func: impl FnMut(&mut CubeContext, <T as CubeType>::ExpandType),
    );
}

pub struct Range<I: Int> {
    pub start: ExpandElementTyped<I>,
    pub end: ExpandElementTyped<I>,
    pub inclusive: bool,
}

impl<I: Int> Range<I> {
    pub fn new(start: ExpandElementTyped<I>, end: ExpandElementTyped<I>, inclusive: bool) -> Self {
        Range {
            start,
            end,
            inclusive,
        }
    }

    pub fn __expand_step_by(self, n: impl Into<ExpandElementTyped<u32>>) -> SteppedRange<I> {
        SteppedRange {
            start: self.start,
            end: self.end,
            step: n.into(),
            inclusive: self.inclusive,
        }
    }
}

impl<I: Int> Iterable<I> for Range<I> {
    fn expand_unroll(
        self,
        context: &mut CubeContext,
        mut func: impl FnMut(&mut CubeContext, <I as CubeType>::ExpandType),
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
                func(context, var.into())
            }
        } else {
            for i in start..end {
                let var = I::from_int(i);
                func(context, var.into())
            }
        }
    }

    fn expand(
        self,
        context: &mut CubeContext,
        mut func: impl FnMut(&mut CubeContext, <I as CubeType>::ExpandType),
    ) {
        let mut child = context.child();
        let index_ty = Item::new(I::as_elem());
        let i = child.scope.borrow_mut().create_local_undeclared(index_ty);
        let i = ExpandElement::Plain(i);

        func(&mut child, i.clone().into());

        context.register(Branch::RangeLoop(RangeLoop {
            i: *i,
            start: *self.start.expand,
            end: *self.end.expand,
            step: None,
            scope: child.into_scope(),
            inclusive: self.inclusive,
        }));
    }
}

pub struct SteppedRange<I: Int> {
    start: ExpandElementTyped<I>,
    end: ExpandElementTyped<I>,
    step: ExpandElementTyped<u32>,
    inclusive: bool,
}

impl<I: Int + Into<ExpandElement>> Iterable<I> for SteppedRange<I> {
    fn expand(
        self,
        context: &mut CubeContext,
        mut func: impl FnMut(&mut CubeContext, <I as CubeType>::ExpandType),
    ) {
        let mut child = context.child();
        let index_ty = Item::new(I::as_elem());
        let i = child.scope.borrow_mut().create_local_undeclared(index_ty);
        let i = ExpandElement::Plain(i);

        func(&mut child, i.clone().into());

        context.register(Branch::RangeLoop(RangeLoop {
            i: *i,
            start: *self.start.expand,
            end: *self.end.expand,
            step: Some(*self.step.expand),
            scope: child.into_scope(),
            inclusive: self.inclusive,
        }));
    }

    fn expand_unroll(
        self,
        context: &mut CubeContext,
        mut func: impl FnMut(&mut CubeContext, <I as CubeType>::ExpandType),
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
                func(context, var.into())
            }
        } else {
            for i in (start..end).step_by(step) {
                let var = I::from_int(i);
                func(context, var.into())
            }
        }
    }
}

/// integer range. Equivalent to:
///
/// ```ignore
/// for i in start..end { ... }
/// ```
pub fn range<T: Int>(start: T, end: T) -> impl Iterator<Item = T> {
    let start: i64 = start.to_i64().unwrap();
    let end: i64 = end.to_i64().unwrap();
    (start..end).map(<T as NumCast>::from).map(Option::unwrap)
}

/// Stepped range. Equivalent to:
///
/// ```ignore
/// for i in (start..end).step_by(step) { ... }
/// ```
pub fn range_stepped<I: Int>(start: I, end: I, step: I) -> impl Iterator<Item = I>
where
    Range<I>: Iterator,
{
    let start = start.to_i64().unwrap();
    let end = end.to_i64().unwrap();
    let step = step.to_usize().unwrap();
    (start..end)
        .step_by(step)
        .map(<I as NumCast>::from)
        .map(Option::unwrap)
}

pub fn for_expand<I: Numeric>(
    context: &mut CubeContext,
    range: impl Iterable<I>,
    unroll: bool,
    func: impl FnMut(&mut CubeContext, ExpandElementTyped<I>),
) {
    if unroll {
        range.expand_unroll(context, func);
    } else {
        range.expand(context, func);
    }
}

pub fn if_expand(
    context: &mut CubeContext,
    runtime_cond: ExpandElement,
    block: impl FnOnce(&mut CubeContext),
) {
    let comptime_cond = runtime_cond.as_const().map(|it| it.as_bool());
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

pub fn if_else_expand(
    context: &mut CubeContext,
    runtime_cond: ExpandElement,
    then_block: impl FnOnce(&mut CubeContext),
    else_block: impl FnOnce(&mut CubeContext),
) {
    let comptime_cond = runtime_cond.as_const().map(|it| it.as_bool());
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

pub fn while_loop_expand<FC, FB>(
    context: &mut CubeContext,
    mut cond_fn: impl FnMut(&mut CubeContext) -> ExpandElementTyped<bool>,
    block: impl FnOnce(&mut CubeContext),
) {
    let mut inside_loop = context.child();

    let cond: ExpandElement = cond_fn(&mut inside_loop).into();
    if_expand(&mut inside_loop, cond, break_expand);

    block(&mut inside_loop);
    context.register(Branch::Loop(Loop {
        scope: inside_loop.into_scope(),
    }));
}

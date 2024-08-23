use crate::prelude::Int;
use std::fmt::Display;

use super::{
    AddExpr, BinaryOp, Block, Expr, Expression, Literal, MethodExpand, SquareType, Variable,
};

pub struct Break;

impl Expr for Break {
    type Output = ();

    fn expression_untyped(&self) -> super::Expression {
        Expression::Break
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

pub struct Continue;

impl Expr for Continue {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::Continue
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

pub struct ForLoop<TNum: SquareType + Int, Range: ForLoopRange<TNum>> {
    pub range: Range,
    pub unroll: bool,
    pub variable: Variable<TNum>,

    pub block: Block<()>,
}

pub trait ForLoopRange<TNum> {
    fn start(&self) -> impl Expr<Output = TNum>;
    fn end(&self) -> impl Expr<Output = TNum>;
    fn step(&self) -> impl Expr<Output = TNum>;
}

impl<TNum: SquareType + Int, Range: ForLoopRange<TNum>> Expr for ForLoop<TNum, Range> {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::ForLoop {
            from: Box::new(self.range.start().expression_untyped()),
            to: Box::new(self.range.end().expression_untyped()),
            step: Box::new(self.range.step().expression_untyped()),
            unroll: self.unroll,
            variable: Box::new(self.variable.expression_untyped()),
            block: self.block.statements.iter().cloned().collect(),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

pub struct RangeExpr<TNum: SquareType + Int, Start: Expr<Output = TNum>, End: Expr<Output = TNum>> {
    pub start: Start,
    pub end: End,
}

impl<TNum: SquareType + Int + Display, Start: Expr<Output = TNum>, End: Expr<Output = TNum>>
    RangeExpr<TNum, Start, End>
{
    pub fn new_exclusive(start: Start, end: End) -> Self {
        RangeExpr { start, end }
    }
}

impl<TNum: SquareType + Int + Display, Start: Expr<Output = TNum>, End: Expr<Output = TNum>>
    RangeExpr<TNum, Start, AddExpr<End, Literal<TNum>, TNum>>
{
    pub fn new_inclusive(start: Start, end: End) -> Self {
        RangeExpr {
            start,
            end: AddExpr(BinaryOp::new(end, Literal::new(TNum::from(1)))),
        }
    }
}

#[derive(new)]
pub struct SteppedRangeExpr<
    TNum: SquareType + Int + Display,
    Start: Expr<Output = TNum>,
    End: Expr<Output = TNum>,
    Step: Expr<Output = TNum>,
    Inner: Expr<Output = RangeExpr<TNum, Start, End>>,
> {
    pub inner: Inner,
    pub step: Step,
}

pub struct RangeExprExpand<
    TNum: SquareType + Int + Display,
    Start: Expr<Output = TNum>,
    End: Expr<Output = TNum>,
    Inner: Expr<Output = RangeExpr<TNum, Start, End>>,
>(Inner);

impl<
        TNum: SquareType + Int + Display,
        Start: Expr<Output = TNum>,
        End: Expr<Output = TNum>,
        Inner: Expr<Output = RangeExpr<TNum, Start, End>>,
    > RangeExprExpand<TNum, Start, End, Inner>
{
    pub fn step_by<Step: Expr<Output = TNum>>(
        self,
        step: Step,
    ) -> SteppedRangeExpr<TNum, Start, End, Step, Inner> {
        SteppedRangeExpr::new(self.0, step)
    }
}

impl<TNum: SquareType + Int + Display, Start: Expr<Output = TNum>, End: Expr<Output = TNum>>
    MethodExpand for RangeExpr<TNum, Start, End>
{
    type Expanded<Inner: Expr<Output = Self>> = RangeExprExpand<TNum, Start, End, Inner>;

    fn expand_methods<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        RangeExprExpand(inner)
    }
}

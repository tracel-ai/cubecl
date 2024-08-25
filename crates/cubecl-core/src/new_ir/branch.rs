use std::num::NonZero;

use super::{Block, Expand, Expr, Expression, Integer, Range, SquareType, TypeEq, Variable};

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

pub trait ForLoopRange {
    type Primitive: SquareType;
}

pub struct ForLoop<Range: Expr>
where
    Range::Output: ForLoopRange,
{
    pub range: Range,
    pub unroll: bool,
    pub variable: Variable<<Range::Output as ForLoopRange>::Primitive>,

    pub block: Block<()>,
}

impl<Range: Expr> ForLoop<Range>
where
    Range::Output: ForLoopRange,
{
    pub fn new(
        range: Range,
        variable: Variable<<Range::Output as ForLoopRange>::Primitive>,
        block: Block<()>,
    ) -> Self {
        Self {
            range,
            variable,
            block,
            unroll: false,
        }
    }
}

impl<Range: Expr> ForLoop<Range>
where
    Range::Output: ForLoopRange,
{
    pub fn new_unroll(
        range: Range,
        variable: Variable<<Range::Output as ForLoopRange>::Primitive>,
        block: Block<()>,
    ) -> Self {
        Self {
            range,
            variable,
            block,
            unroll: true,
        }
    }
}

impl<Range: Expr> Expr for ForLoop<Range>
where
    Range::Output: ForLoopRange,
{
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        let range = self.range.expression_untyped().as_range().unwrap().clone();
        if self.unroll {
            assert!(
                matches!(*range.start, Expression::Literal { .. }),
                "Can't unroll loop with dynamic start"
            );
            assert!(
                matches!(*range.end, Expression::Literal { .. }),
                "Can't unroll loop with dynamic end"
            );
            if let Some(step) = &range.step {
                assert!(
                    matches!(**step, Expression::Literal { .. }),
                    "Can't unroll loop with dynamic step"
                );
            }
        }
        Expression::ForLoop {
            range,
            unroll: self.unroll,
            variable: Box::new(self.variable.expression_untyped()),
            block: Box::new(self.block.expression_untyped()),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct RangeExpr<Start: Expr, End: Expr>
where
    Start::Output: SquareType + TypeEq<End::Output>,
{
    pub start: Start,
    pub end: End,
    pub inclusive: bool,
}

#[derive(new)]
pub struct SteppedRangeExpr<Start: Expr, End: Expr, Step: Expr, Inner>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
    End::Output: TypeEq<Step::Output>,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    pub inner: Inner,
    pub step: Step,
}

pub struct RangeExprExpand<Start: Expr, End: Expr, Inner>(Inner)
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
    Inner: Expr<Output = RangeExpr<Start, End>>;

impl<Start: Expr, End: Expr, Inner> RangeExprExpand<Start, End, Inner>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    pub fn step_by<Step: Expr>(self, step: Step) -> SteppedRangeExpr<Start, End, Step, Inner>
    where
        End::Output: TypeEq<Step::Output>,
    {
        SteppedRangeExpr::new(self.0, step)
    }
}

impl<Start: Expr, End: Expr> Expand for RangeExpr<Start, End>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
{
    type Expanded<Inner: Expr<Output = Self>> = RangeExprExpand<Start, End, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        RangeExprExpand(inner)
    }
}

impl<Start: Expr, End: Expr> Expr for RangeExpr<Start, End>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
{
    type Output = Self;

    fn expression_untyped(&self) -> Expression {
        Expression::__Range(Range {
            start: Box::new(self.start.expression_untyped()),
            end: Box::new(self.end.expression_untyped()),
            step: None,
            inclusive: self.inclusive,
        })
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<Start: Expr, End: Expr> ForLoopRange for RangeExpr<Start, End>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
{
    type Primitive = Start::Output;
}

impl<Start: Expr, End: Expr, Step: Expr, Inner> Expr for SteppedRangeExpr<Start, End, Step, Inner>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
    End::Output: TypeEq<Step::Output>,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    type Output = Self;

    fn expression_untyped(&self) -> Expression {
        let inner = self.inner.expression_untyped().as_range().unwrap().clone();
        Expression::__Range(Range {
            step: Some(Box::new(self.step.expression_untyped())),
            ..inner
        })
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<Start: Expr, End: Expr, Step: Expr, Inner> ForLoopRange
    for SteppedRangeExpr<Start, End, Step, Inner>
where
    Start::Output: SquareType + Integer + TypeEq<End::Output>,
    End::Output: TypeEq<Step::Output>,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    type Primitive = Start::Output;
}

#[derive(new)]
pub struct WhileLoop<Condition: Expr<Output = bool>> {
    pub condition: Condition,
    pub block: Block<()>,
}

impl<Condition: Expr<Output = bool>> Expr for WhileLoop<Condition> {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::WhileLoop {
            condition: Box::new(self.condition.expression_untyped()),
            block: Box::new(self.block.expression_untyped()),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Loop(pub Block<()>);

impl Expr for Loop {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::Loop {
            block: Box::new(self.0.expression_untyped()),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct If<Condition: Expr<Output = bool>, OutIf: Expr = (), OutElse: Expr = ()>
where
    OutIf::Output: SquareType + TypeEq<OutElse::Output>,
    OutElse::Output: SquareType,
{
    pub condition: Condition,
    pub then_block: Block<OutIf>,
    pub else_branch: Option<OutElse>,
}

impl<Condition: Expr<Output = bool>, OutIf: Expr, OutElse: Expr> Expr
    for If<Condition, OutIf, OutElse>
where
    OutIf::Output: SquareType + TypeEq<OutElse::Output>,
    OutElse::Output: SquareType,
{
    type Output = OutIf::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::If {
            condition: Box::new(self.condition.expression_untyped()),
            then_block: Box::new(self.then_block.expression_untyped()),
            else_branch: self
                .else_branch
                .as_ref()
                .map(|it| it.expression_untyped())
                .map(Box::new),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Return<Type: SquareType, Ret: Expr<Output = Type>>(pub Option<Ret>);

impl<Type: SquareType, Ret: Expr<Output = Type>> Expr for Return<Type, Ret> {
    type Output = Ret;

    fn expression_untyped(&self) -> Expression {
        Expression::Return {
            expr: self
                .0
                .as_ref()
                .map(|it| it.expression_untyped())
                .map(Box::new),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

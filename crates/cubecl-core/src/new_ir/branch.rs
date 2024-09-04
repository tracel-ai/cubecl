use super::{BlockExpr, Expand, Expanded, Expr, Expression, Range, SquareType, Variable};
use crate::prelude::Integer;
use std::{num::NonZero, rc::Rc};

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
    type Primitive: Integer;

    //fn as_primitive(&self) -> (i64, i64, Option<i64>, bool);
}

pub struct ForLoop<Range: Expr>
where
    Range::Output: ForLoopRange,
{
    pub range: Range,
    pub unroll: bool,
    pub variable: Variable<<Range::Output as ForLoopRange>::Primitive>,

    pub block: Rc<BlockExpr<()>>,
}

impl<Range: Expr> ForLoop<Range>
where
    Range::Output: ForLoopRange,
{
    pub fn new(
        range: Range,
        variable: Variable<<Range::Output as ForLoopRange>::Primitive>,
        block: BlockExpr<()>,
    ) -> Self {
        Self {
            range,
            variable,
            block: Rc::new(block),
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
        block: BlockExpr<()>,
    ) -> Self {
        Self {
            range,
            variable,
            block: Rc::new(block),
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
            variable: self.variable.expression_untyped().as_variable().unwrap(),
            block: self.block.expression_untyped().as_block().unwrap(),
            unroll: self.unroll,
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct RangeExpr<Start: Expr, End: Expr<Output = Start::Output>>
where
    Start::Output: Integer,
{
    pub start: Start,
    pub end: End,
    pub inclusive: bool,
}

#[derive(new)]
pub struct SteppedRangeExpr<
    Start: Expr,
    End: Expr<Output = Start::Output>,
    Step: Expr<Output = Start::Output>,
    Inner,
> where
    Start::Output: Integer,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    pub inner: Inner,
    pub step: Step,
}

pub struct RangeExprExpand<Start: Expr, End: Expr<Output = Start::Output>, Inner>(Inner)
where
    Start::Output: Integer,
    Inner: Expr<Output = RangeExpr<Start, End>>;

impl<Start: Expr, End: Expr<Output = Start::Output>, Inner> Expanded
    for RangeExprExpand<Start, End, Inner>
where
    Start::Output: Integer,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    type Unexpanded = RangeExpr<Start, End>;

    fn inner(self) -> impl Expr<Output = Self::Unexpanded> {
        self.0
    }
}

impl<Start: Expr, End: Expr<Output = Start::Output>, Inner> RangeExprExpand<Start, End, Inner>
where
    Start::Output: SquareType + Integer,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    pub fn step_by<Step: Expr<Output = Start::Output>>(
        self,
        step: Step,
    ) -> SteppedRangeExpr<Start, End, Step, Inner> {
        SteppedRangeExpr::new(self.0, step)
    }
}

impl<Start: Expr, End: Expr<Output = Start::Output>> Expand for RangeExpr<Start, End>
where
    Start::Output: Integer,
{
    type Expanded<Inner: Expr<Output = Self>> = RangeExprExpand<Start, End, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        RangeExprExpand(inner)
    }
}

impl<Start: Expr, End: Expr<Output = Start::Output>> Expr for RangeExpr<Start, End>
where
    Start::Output: Integer,
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

impl<Start: Expr, End: Expr<Output = Start::Output>> ForLoopRange for RangeExpr<Start, End>
where
    Start::Output: Integer,
{
    type Primitive = Start::Output;

    // fn as_primitive(&self) -> (i64, i64, Option<i64>, bool) {
    //     let start = self.start.expression_untyped();
    //     let end = self.end.expression_untyped();
    //     assert!(
    //         matches!(start, Expression::Literal { .. }),
    //         "Can't unroll loop with dynamic start"
    //     );
    //     assert!(
    //         matches!(end, Expression::Literal { .. }),
    //         "Can't unroll loop with dynamic end"
    //     );
    //     let start = start.as_lit().unwrap();
    //     let end = end.as_lit().unwrap();
    //     match start {
    //         ConstantScalarValue::Int(i, _) => (i, end.as_i64(), None, self.inclusive),
    //         ConstantScalarValue::UInt(u) => (u as i64, end.as_u64() as i64, None, self.inclusive),
    //         _ => unreachable!(),
    //     }
    // }
}

impl<Start: Expr, End: Expr<Output = Start::Output>, Step: Expr<Output = Start::Output>, Inner> Expr
    for SteppedRangeExpr<Start, End, Step, Inner>
where
    Start::Output: Integer,
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

impl<Start: Expr, End: Expr<Output = Start::Output>, Step: Expr<Output = Start::Output>, Inner>
    ForLoopRange for SteppedRangeExpr<Start, End, Step, Inner>
where
    Start::Output: Integer,
    Inner: Expr<Output = RangeExpr<Start, End>>,
{
    type Primitive = Start::Output;

    // fn as_primitive(&self) -> (i64, i64, Option<i64>, bool) {
    //     let inner = self.inner.expression_untyped();
    //     let inner = inner.as_range().unwrap().clone();
    //     let step = self.step.expression_untyped();
    //     assert!(
    //         matches!(*inner.start, Expression::Literal { .. }),
    //         "Can't unroll loop with dynamic start"
    //     );
    //     assert!(
    //         matches!(*inner.end, Expression::Literal { .. }),
    //         "Can't unroll loop with dynamic end"
    //     );
    //     assert!(
    //         matches!(step, Expression::Literal { .. }),
    //         "Can't unroll loop with dynamic step"
    //     );
    //     let start = inner.start.as_lit().unwrap();
    //     let end = inner.end.as_lit().unwrap();
    //     let step = step.as_lit().unwrap();
    //     match step {
    //         ConstantScalarValue::Int(i, _) => {
    //             (start.as_i64(), end.as_i64(), Some(i), inner.inclusive)
    //         }
    //         ConstantScalarValue::UInt(u) => (
    //             start.as_u64() as i64,
    //             end.as_u64() as i64,
    //             Some(u as i64),
    //             inner.inclusive,
    //         ),
    //         _ => unreachable!(),
    //     }
    // }
}

#[derive(new)]
pub struct WhileLoop<Condition: Expr<Output = bool>> {
    pub condition: Condition,
    pub block: BlockExpr<()>,
}

impl<Condition: Expr<Output = bool>> Expr for WhileLoop<Condition> {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::WhileLoop {
            condition: Box::new(self.condition.expression_untyped()),
            block: self.block.expression_untyped().as_block().unwrap(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct Loop(pub BlockExpr<()>);

impl Expr for Loop {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::Loop {
            block: self.0.expression_untyped().as_block().unwrap(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

#[derive(new)]
pub struct If<
    Condition: Expr<Output = bool>,
    OutIf: Expr = (),
    OutElse: Expr<Output = OutIf::Output> = (),
> where
    OutIf::Output: SquareType,
{
    pub condition: Condition,
    pub then_block: BlockExpr<OutIf>,
    pub else_branch: Option<OutElse>,
}

impl<Condition: Expr<Output = bool>, OutIf: Expr, OutElse: Expr<Output = OutIf::Output>> Expr
    for If<Condition, OutIf, OutElse>
where
    OutIf::Output: SquareType,
{
    type Output = OutIf::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::If {
            condition: Box::new(self.condition.expression_untyped()),
            then_block: self.then_block.expression_untyped().as_block().unwrap(),
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
pub struct Return<Type: SquareType = (), Ret: Expr<Output = Type> = ()>(pub Option<Ret>);

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

use super::{Block, Expr, Expression, SquareType, Variable};

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

pub struct ForLoop<TNum: SquareType> {
    pub from: Box<dyn Expr<Output = TNum>>,
    pub to: Box<dyn Expr<Output = TNum>>,
    pub step: Option<Box<dyn Expr<Output = TNum>>>,
    pub unroll: bool,
    pub variable: Variable<TNum>,

    pub block: Block<()>,
}

impl<TNum: SquareType> Expr for ForLoop<TNum> {
    type Output = ();

    fn expression_untyped(&self) -> Expression {
        Expression::ForLoop {
            from: Box::new(self.from.expression_untyped()),
            to: Box::new(self.to.expression_untyped()),
            step: self
                .step
                .as_ref()
                .map(|step| Box::new(step.expression_untyped())),
            unroll: self.unroll,
            variable: Box::new(self.variable.expression_untyped()),
            block: self.block.statements.iter().cloned().collect(),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

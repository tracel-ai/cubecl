use super::{Elem, Expr, Expression, Primitive, SquareType, UnaryOp, Vectorization};

#[derive(Clone, Debug, PartialEq)]
pub enum SubcubeExpression {
    Elect,
    Broadcast {
        left: Box<Expression>,
        right: Box<Expression>,
        ty: Elem,
        vectorization: Vectorization,
    },
    Unary {
        input: Box<Expression>,
        operation: SubcubeOp,
        ty: Elem,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum SubcubeOp {
    All,
    Any,
    Sum,
    Prod,
    And,
    Or,
    Xor,
    Min,
    Max,
}

impl SubcubeExpression {
    pub fn ir_type(&self) -> Elem {
        match self {
            SubcubeExpression::Elect => Elem::Bool,
            SubcubeExpression::Broadcast { ty, .. } => *ty,
            SubcubeExpression::Unary { ty, .. } => *ty,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        match self {
            SubcubeExpression::Elect => None,
            SubcubeExpression::Broadcast { vectorization, .. } => *vectorization,
            SubcubeExpression::Unary { input, .. } => input.vectorization(),
        }
    }
}

macro_rules! unary_op {
    ($name:ident, $op:ident) => {
        pub struct $name<In: Expr>(UnaryOp<In, In::Output>)
        where
            In::Output: Primitive;

        impl<In: Expr> $name<In>
        where
            In::Output: Primitive,
        {
            pub fn new(input: In) -> Self {
                Self(UnaryOp::new(input))
            }
        }

        impl<In: Expr> Expr for $name<In>
        where
            In::Output: Primitive,
        {
            type Output = In::Output;

            fn expression_untyped(&self) -> Expression {
                SubcubeExpression::Unary {
                    input: Box::new(self.0.input.expression_untyped()),
                    ty: <In::Output as SquareType>::ir_type(),
                    operation: SubcubeOp::$op,
                }
                .into()
            }

            fn vectorization(&self) -> Vectorization {
                self.0.input.vectorization()
            }
        }
    };
}

unary_op!(SubcubeSumExpr, Sum);
unary_op!(SubcubeProdExpr, Prod);
unary_op!(SubcubeMaxExpr, Max);
unary_op!(SubcubeMinExpr, Min);
unary_op!(SubcubeAllExpr, All);
unary_op!(SubcubeAnyExpr, Any);
unary_op!(SubcubeAndExpr, And);
unary_op!(SubcubeOrExpr, Or);
unary_op!(SubcubeXorExpr, Xor);

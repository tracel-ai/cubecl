use super::{element::Array, Expr, Expression, Integer, Primitive};

#[derive(new)]
pub struct ArrayInit<Init: Expr, Size: Expr>
where
    Init::Output: Primitive,
    Size::Output: Integer,
{
    pub size: Size,
    pub init: Init,
}

impl<Init: Expr, Size: Expr> Expr for ArrayInit<Init, Size>
where
    Init::Output: Primitive,
    Size::Output: Integer,
{
    type Output = Array<Init::Output>;

    fn expression_untyped(&self) -> super::Expression {
        Expression::ArrayInit {
            size: Box::new(self.size.expression_untyped()),
            init: Box::new(self.init.expression_untyped()),
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.init.vectorization()
    }
}

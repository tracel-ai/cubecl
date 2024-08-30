use cubecl_core::{ir::Elem, new_ir::*};
use cubecl_macros_2::{cube2, expand_impl, Expand};
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[cube2]
fn helper_fn(a: u32) -> u32 {
    a * 2
}

#[test]
fn function_call() {
    #[allow(unused)]
    #[cube2]
    fn function_call(a: u32) -> u32 {
        helper_fn(a)
    }

    let expanded = function_call::expand(Variable::new("a", None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(block_expr(
            vec![],
            Some(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::Mul,
                right: Box::new(lit(2u32)),
                vectorization: None,
                ty: Elem::UInt,
            }),
        )),
    );

    assert_eq!(expanded, expected);
}

#[derive(Expand)]
struct Dummy {
    a: u32,
}

#[expand_impl]
impl Dummy {
    fn method(&self, b: u32) -> u32 {
        self.a * b
    }

    #[expanded]
    pub fn method<B: Expr<Output = u32>>(self, b: B) -> impl Expr<Output = u32> {
        MulExpr::new(self.0.expand().__a(), b)
    }
}

#[test]
fn method_call() {
    #[allow(unused)]
    #[cube2]
    fn method_call(a: Dummy) -> u32 {
        a.method(2)
    }

    let expanded = method_call::expand(Variable::new("a", None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Binary {
            left: Box::new(Expression::FieldAccess {
                base: var_expr("a", Elem::Unit),
                name: "a".to_string(),
                vectorization: None,
                ty: Elem::UInt,
            }),
            operator: Operator::Mul,
            right: Box::new(lit(2u32)),
            vectorization: None,
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

#[expand_impl]
impl Dummy {
    fn associated(b: u32) -> u32 {
        b * 2
    }

    #[expanded]
    pub fn associated<B: Expr<Output = u32>>(b: B) -> impl Expr<Output = u32> {
        MulExpr::new(b, 2)
    }
}

#[test]
fn associated_call() {
    #[allow(unused)]
    #[cube2]
    fn associated_call() -> u32 {
        Dummy::associated(4)
    }

    let expanded = associated_call::expand().expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Binary {
            left: Box::new(lit(4u32)),
            operator: Operator::Mul,
            right: Box::new(lit(2u32)),
            vectorization: None,
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

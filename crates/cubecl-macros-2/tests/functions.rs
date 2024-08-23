use cubecl_core::{
    ir::Elem,
    new_ir::{
        BinaryOp, Block, Expr, Expression, FieldExpandExpr, MethodExpand, MethodExpandExpr,
        MulExpr, Operator, Statement, Variable,
    },
};
use cubecl_macros_2::{cube2, KernelArg};
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

    let expanded = function_call::expand(Variable::new("a", None));
    let expected = Block::<u32>::new(vec![Statement::Return(Box::new(Expression::Block {
        inner: vec![],
        ret: Some(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            operator: Operator::Mul,
            right: lit(2u32),
            vectorization: None,
            ty: Elem::UInt,
        })),
        vectorization: None,
        ty: Elem::UInt,
    }))]);

    assert_eq!(expanded, expected);
}
#[derive(KernelArg)]
struct Dummy {
    a: u32,
}

impl Dummy {
    fn method(&self, b: u32) -> u32 {
        self.a * b
    }
}

struct DummyMethods<E: Expr<Output = Dummy>>(E);

impl<E: Expr<Output = Dummy>> DummyMethods<E> {
    pub fn method<B: Expr<Output = u32>>(self, b: B) -> impl Expr<Output = u32> {
        MulExpr(BinaryOp::new(self.0.expand_fields().field_a(), b))
    }
}

impl MethodExpand for Dummy {
    type Expanded<Inner: Expr<Output = Self>> = DummyMethods<Inner>;

    fn expand_methods<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        DummyMethods(inner)
    }
}

#[test]
fn method_call() {
    #[allow(unused)]
    #[cube2]
    fn method_call(a: Dummy) -> u32 {
        a.method(2)
    }

    let expanded = method_call::expand(Variable::new("a", None));
    let expected = Block::<u32>::new(vec![Statement::Return(Box::new(Expression::Binary {
        left: Box::new(Expression::FieldAccess {
            base: var("a", Elem::Pointer),
            name: "a".to_string(),
            vectorization: None,
            ty: Elem::UInt,
        }),
        operator: Operator::Mul,
        right: lit(2u32),
        vectorization: None,
        ty: Elem::UInt,
    }))]);

    assert_eq!(expanded, expected);
}

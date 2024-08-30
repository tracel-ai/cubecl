use common::*;
use cubecl_core::{
    ir::Elem,
    new_ir::{Expr, Expression, TensorExpression},
};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;

#[test]
fn array_init() {
    #[allow(unused)]
    #[cube2]
    fn array_init() -> u32 {
        let local = [2; 10];
        local[2]
    }

    let expanded = array_init::expand().expression_untyped();
    let expected = Expression::Block(block(
        vec![local_init(
            "local",
            Expression::ArrayInit {
                size: Box::new(lit(10)),
                init: Box::new(lit(2u32)),
            },
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("local", Elem::UInt),
            index: Box::new(lit(2)),
        })),
    ));

    assert_eq!(expanded, expected);
}

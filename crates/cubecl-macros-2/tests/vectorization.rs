use std::num::NonZero;

use cubecl_core::{
    ir::Elem,
    new_ir::{Block, Expression, Operator, Statement, Variable},
};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[test]
pub fn vectorization_simple() {
    #[allow(unused)]
    #[cube2]
    fn vectorized(a: u32, b: u32) -> u32 {
        let c = a * b; // a = vec4(u32), b = u32, c = vec4(u32)
        c * a // return = vec4(u32) * vec4(u32)
    }

    let expanded = vectorized::expand(
        Variable::new("a", NonZero::new(4)),
        Variable::new("b", None),
    );
    let expected = Block::<u32>::new(vec![
        init_vec(
            "c",
            Expression::Binary {
                left: Box::new(vec_var("a", Elem::UInt, 4)),
                operator: Operator::Mul,
                right: var("b", Elem::UInt),
                vectorization: NonZero::new(4),
                ty: Elem::UInt,
            },
            false,
            None,
            4,
        ),
        Statement::Return(Expression::Binary {
            left: Box::new(vec_var("c", Elem::UInt, 4)),
            operator: Operator::Mul,
            right: Box::new(vec_var("a", Elem::UInt, 4)),
            vectorization: NonZero::new(4),
            ty: Elem::UInt,
        }),
    ]);

    assert_eq!(expanded, expected);
}

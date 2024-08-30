#![allow(clippy::all)]

mod common;
use common::*;
use cubecl_core::{
    ir::{Elem, FloatKind, IntKind},
    new_ir::{Expr, Expression, Operator},
};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;
use Expression::Binary;

#[test]
fn simple_arithmetic() {
    #[allow(unused)]
    #[cube2]
    fn simple_arithmetic() {
        let mut a: u32 = 1;
        let mut b = a * 3;
        let mut c = b + a;
        let mut d = 2 / a;
        let mut e = 3 % b;
        let mut f = b - a;
    }

    let expansion = simple_arithmetic::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(1u32), true, Some(Elem::UInt)),
            local_init(
                "b",
                Expression::Binary {
                    left: var_expr("a", Elem::UInt),
                    right: Box::new(lit(3u32)),
                    operator: Operator::Mul,
                    ty: Elem::UInt,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "c",
                Expression::Binary {
                    left: var_expr("b", Elem::UInt),
                    operator: Operator::Add,
                    right: var_expr("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "d",
                Expression::Binary {
                    left: Box::new(lit(2u32)),
                    operator: Operator::Div,
                    right: var_expr("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "e",
                Expression::Binary {
                    left: Box::new(lit(3u32)),
                    operator: Operator::Rem,
                    right: var_expr("b", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "f",
                Expression::Binary {
                    left: var_expr("b", Elem::UInt),
                    operator: Operator::Sub,
                    right: var_expr("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                },
                true,
                None,
            ),
        ],
        None,
    );

    assert_eq!(expansion, expected);
}

#[test]
fn cmp_ops() {
    #[allow(unused)]
    #[cube2]
    fn cmp_ops() {
        let mut a = 1u32;
        let mut b = a > 1u32;
        let mut c = a <= 1u32;
        let mut d = a < 11u32;
        let mut e = 1u32 >= a;
        let mut f = a == 2u32;
        let mut g = a != 2u32;
    }

    let expanded = cmp_ops::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(1u32), true, None),
            local_init(
                "b",
                Binary {
                    left: var_expr("a", Elem::UInt),
                    operator: Operator::Gt,
                    right: Box::new(lit(1u32)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "c",
                Binary {
                    left: var_expr("a", Elem::UInt),
                    operator: Operator::Le,
                    right: Box::new(lit(1u32)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "d",
                Binary {
                    left: var_expr("a", Elem::UInt),
                    operator: Operator::Lt,
                    right: Box::new(lit(11u32)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "e",
                Binary {
                    left: Box::new(lit(1u32)),
                    operator: Operator::Ge,
                    right: var_expr("a", Elem::UInt),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "f",
                Binary {
                    left: var_expr("a", Elem::UInt),
                    operator: Operator::Eq,
                    right: Box::new(lit(2u32)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init(
                "g",
                Binary {
                    left: var_expr("a", Elem::UInt),
                    operator: Operator::Ne,
                    right: Box::new(lit(2u32)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
        ],
        None,
    );

    assert_eq!(expanded, expected);
}

#[test]
fn assign_arithmetic() {
    #[allow(unused)]
    #[cube2]
    fn assign_arithmetic() {
        let mut a: u32 = 1;
        a *= 3;
        a += 2;
        a /= 2;
        a %= 1;
        a -= 0;
    }

    let expansion = assign_arithmetic::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(1u32), true, Some(Elem::UInt)),
            expr(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                right: Box::new(lit(3u32)),
                operator: Operator::MulAssign,
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::AddAssign,
                right: Box::new(lit(2u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::DivAssign,
                right: Box::new(lit(2u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::RemAssign,
                right: Box::new(lit(1u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Expression::Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::SubAssign,
                right: Box::new(lit(0u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
        ],
        None,
    );

    assert_eq!(expansion, expected);
}

#[test]
fn boolean_ops() {
    #[allow(unused)]
    #[cube2]
    fn bool_ops() {
        let mut a = false;
        let mut b = a && true;
        let mut c = 1;
        b || a;
        c ^ 2;
        c | 3;
        c & 1;
    }

    let expanded = bool_ops::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(false), true, None),
            local_init(
                "b",
                Binary {
                    left: var_expr("a", Elem::Bool),
                    operator: Operator::And,
                    right: Box::new(lit(true)),
                    ty: Elem::Bool,
                    vectorization: None,
                },
                true,
                None,
            ),
            local_init("c", lit(1), true, None),
            expr(Binary {
                left: var_expr("b", Elem::Bool),
                operator: Operator::Or,
                right: var_expr("a", Elem::Bool),
                ty: Elem::Bool,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("c", Elem::Int(IntKind::I32)),
                operator: Operator::BitXor,
                right: Box::new(lit(2)),
                ty: Elem::Int(IntKind::I32),
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("c", Elem::Int(IntKind::I32)),
                operator: Operator::BitOr,
                right: Box::new(lit(3)),
                ty: Elem::Int(IntKind::I32),
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("c", Elem::Int(IntKind::I32)),
                operator: Operator::BitAnd,
                right: Box::new(lit(1)),
                ty: Elem::Int(IntKind::I32),
                vectorization: None,
            }),
        ],
        None,
    );

    assert_eq!(expanded, expected);
}

#[test]
fn boolean_assign_ops() {
    #[allow(unused)]
    #[cube2]
    fn bool_assign_ops() {
        let mut a = 10u32;
        a |= 5;
        a &= 10;
        a ^= 3;
    }

    let expanded = bool_assign_ops::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(10u32), true, None),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::BitOrAssign,
                right: Box::new(lit(5u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::BitAndAssign,
                right: Box::new(lit(10u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::BitXorAssign,
                right: Box::new(lit(3u32)),
                ty: Elem::UInt,
                vectorization: None,
            }),
        ],
        None,
    );

    assert_eq!(expanded, expected);
}

#[test]
fn shift_ops() {
    #[allow(unused)]
    #[cube2]
    fn shift_ops() {
        let mut a = 10u32;
        a << 5;
        a >> 2;
        a <<= 1;
        a >>= 2;
    }

    let expanded = shift_ops::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(10u32), true, None),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::Shl,
                right: Box::new(lit(5)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::Shr,
                right: Box::new(lit(2)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::ShlAssign,
                right: Box::new(lit(1)),
                ty: Elem::UInt,
                vectorization: None,
            }),
            expr(Binary {
                left: var_expr("a", Elem::UInt),
                operator: Operator::ShrAssign,
                right: Box::new(lit(2)),
                ty: Elem::UInt,
                vectorization: None,
            }),
        ],
        None,
    );

    assert_eq!(expanded, expected);
}

#[test]
fn unary_ops() {
    #[allow(unused)]
    #[cube2]
    fn unary_ops() {
        !true;
        -1.0;
    }

    let expanded = unary_ops::expand().expression_untyped();
    let expected = block_expr(
        vec![
            expr(Expression::Unary {
                input: Box::new(lit(true)),
                operator: Operator::Not,
                ty: Elem::Bool,
                vectorization: None,
            }),
            expr(Expression::Unary {
                input: Box::new(lit(1.0)),
                operator: Operator::Neg,
                ty: Elem::Float(FloatKind::F64),
                vectorization: None,
            }),
        ],
        None,
    );

    assert_eq!(expanded, expected);
}

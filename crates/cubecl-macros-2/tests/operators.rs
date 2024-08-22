mod common;
use std::marker::PhantomData;

use common::*;
use cubecl_core::{
    ir::{Elem, FloatKind, IntKind},
    new_ir::{Block, Expression, Operator},
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

    let expansion = simple_arithmetic::expand();
    let expected = Block::<()> {
        statements: vec![
            local_init("a", lit(1u32), true, Some(Elem::UInt)),
            local_init(
                "b",
                Box::new(Expression::Binary {
                    left: var("a", Elem::UInt),
                    right: lit(3u32),
                    operator: Operator::Mul,
                    ty: Elem::UInt,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "c",
                Box::new(Expression::Binary {
                    left: var("b", Elem::UInt),
                    operator: Operator::Add,
                    right: var("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "d",
                Box::new(Expression::Binary {
                    left: lit(2u32),
                    operator: Operator::Div,
                    right: var("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "e",
                Box::new(Expression::Binary {
                    left: lit(3u32),
                    operator: Operator::Rem,
                    right: var("b", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "f",
                Box::new(Expression::Binary {
                    left: var("b", Elem::UInt),
                    operator: Operator::Sub,
                    right: var("a", Elem::UInt),
                    ty: Elem::UInt,
                    vectorization: None,
                }),
                true,
                None,
            ),
        ],
        _ty: PhantomData,
    };

    assert_eq!(expansion, expected);
}

#[test]
fn cmp_ops() {
    #[allow(unused)]
    #[cube2]
    fn cmp_ops() {
        let mut a = 1u32;
        let mut b = a > 1;
        let mut c = a <= 1;
        let mut d = a < 11;
        let mut e = 1 >= a;
        let mut f = a == 2;
        let mut g = a != 2;
    }

    let expanded = cmp_ops::expand();
    let expected = Block::<()> {
        statements: vec![
            local_init("a", lit(1u32), true, None),
            local_init(
                "b",
                Box::new(Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::Gt,
                    right: lit(1u32),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "c",
                Box::new(Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::Le,
                    right: lit(1u32),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "d",
                Box::new(Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::Lt,
                    right: lit(11u32),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "e",
                Box::new(Binary {
                    left: lit(1u32),
                    operator: Operator::Ge,
                    right: var("a", Elem::UInt),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "f",
                Box::new(Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::Eq,
                    right: lit(2u32),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
            local_init(
                "g",
                Box::new(Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::Ne,
                    right: lit(2u32),
                    ty: Elem::Bool,
                    vectorization: None,
                }),
                true,
                None,
            ),
        ],
        _ty: PhantomData,
    };

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

    let expansion = assign_arithmetic::expand();
    let expected = Block::<()>::new(vec![
        local_init("a", lit(1u32), true, Some(Elem::UInt)),
        expr(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            right: lit(3u32),
            operator: Operator::MulAssign,
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            operator: Operator::AddAssign,
            right: lit(2u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            operator: Operator::DivAssign,
            right: lit(2u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            operator: Operator::RemAssign,
            right: lit(1u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Expression::Binary {
            left: var("a", Elem::UInt),
            operator: Operator::SubAssign,
            right: lit(0u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
    ]);

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

    let expanded = bool_ops::expand();
    let expected = Block::<()>::new(vec![
        local_init("a", lit(false), true, None),
        local_init(
            "b",
            Box::new(Binary {
                left: var("a", Elem::Bool),
                operator: Operator::And,
                right: lit(true),
                ty: Elem::Bool,
                vectorization: None,
            }),
            true,
            None,
        ),
        local_init("c", lit(1), true, None),
        expr(Box::new(Binary {
            left: var("b", Elem::Bool),
            operator: Operator::Or,
            right: var("a", Elem::Bool),
            ty: Elem::Bool,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("c", Elem::Int(IntKind::I32)),
            operator: Operator::BitXor,
            right: lit(2),
            ty: Elem::Int(IntKind::I32),
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("c", Elem::Int(IntKind::I32)),
            operator: Operator::BitOr,
            right: lit(3),
            ty: Elem::Int(IntKind::I32),
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("c", Elem::Int(IntKind::I32)),
            operator: Operator::BitAnd,
            right: lit(1),
            ty: Elem::Int(IntKind::I32),
            vectorization: None,
        })),
    ]);

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

    let expanded = bool_assign_ops::expand();
    let expected = Block::<()>::new(vec![
        local_init("a", lit(10u32), true, None),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::BitOrAssign,
            right: lit(5u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::BitAndAssign,
            right: lit(10u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::BitXorAssign,
            right: lit(3u32),
            ty: Elem::UInt,
            vectorization: None,
        })),
    ]);

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

    let expanded = shift_ops::expand();
    let expected = Block::<()>::new(vec![
        local_init("a", lit(10u32), true, None),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::Shl,
            right: lit(5),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::Shr,
            right: lit(2),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::ShlAssign,
            right: lit(1),
            ty: Elem::UInt,
            vectorization: None,
        })),
        expr(Box::new(Binary {
            left: var("a", Elem::UInt),
            operator: Operator::ShrAssign,
            right: lit(2),
            ty: Elem::UInt,
            vectorization: None,
        })),
    ]);

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

    let expanded = unary_ops::expand();
    let expected = Block::<()>::new(vec![
        expr(Box::new(Expression::Unary {
            input: lit(true),
            operator: Operator::Not,
            ty: Elem::Bool,
            vectorization: None,
        })),
        expr(Box::new(Expression::Unary {
            input: lit(1.0),
            operator: Operator::Neg,
            ty: Elem::Float(FloatKind::F64),
            vectorization: None,
        })),
    ]);

    assert_eq!(expanded, expected);
}

use std::marker::PhantomData;

use cubecl_core::{
    ir::Elem,
    new_ir::{Block, Expression, Operator, Statement, Variable},
};
use cubecl_macros_2::{cube2, KernelArg};
use pretty_assertions::assert_eq;
use Elem::UInt;

mod common;
use common::*;

#[test]
pub fn const_param() {
    #[allow(unused)]
    #[cube2]
    fn const_param(a: u32, #[comptime] b: u32) {
        a * b;
    }

    // Should fail (compile tests not working for me rn).
    // let block = const_param::expand(
    //     Variable::<u32> {
    //         name: "a",
    //         _type: PhantomData,
    //     },
    //     Variable::<u32> {
    //         name: "b",
    //         _type: PhantomData,
    //     },
    // );

    let expanded = const_param::expand(
        Variable::<u32> {
            name: "a",
            vectorization: None,
            _type: PhantomData,
        },
        2,
    );

    let expected = Block::<()> {
        _ty: PhantomData,
        statements: vec![expr(Box::new(Expression::Binary {
            left: var("a", UInt),
            operator: Operator::Mul,
            right: lit(2u32),
            ty: UInt,
            vectorization: None,
        }))],
    };

    assert_eq!(expanded, expected);
}

#[test]
pub fn const_generic() {
    #[allow(unused)]
    #[cube2]
    fn const_generic<const D: u32>(a: u32, #[comptime] b: u32) {
        a * b + D;
    }

    let expanded = const_generic::expand::<3>(
        Variable::<u32> {
            name: "a",
            vectorization: None,
            _type: PhantomData,
        },
        2,
    );

    let expected = Block::<()>::new(vec![expr(Box::new(Expression::Binary {
        left: Box::new(Expression::Binary {
            left: var("a", UInt),
            operator: Operator::Mul,
            right: lit(2u32),
            ty: UInt,
            vectorization: None,
        }),
        operator: Operator::Add,
        right: lit(3u32),
        ty: Elem::UInt,
        vectorization: None,
    }))]);

    assert_eq!(expanded, expected);
}

#[derive(KernelArg)]
struct Param {
    a: u32,
    b: u32,
}

#[test]
pub fn struct_param() {
    #[allow(unused)]
    #[cube2]
    fn struct_param(arg: Param) -> u32 {
        arg.a * arg.b
    }

    let expanded = struct_param::expand(Variable::new("param", None));
    let expected = Block::<u32>::new(vec![Statement::Return(Box::new(Expression::Binary {
        left: Box::new(Expression::FieldAccess {
            base: var("param", Elem::Pointer),
            name: "a".to_string(),
            ty: Elem::UInt,
            vectorization: None,
        }),
        operator: Operator::Mul,
        right: Box::new(Expression::FieldAccess {
            base: var("param", Elem::Pointer),
            name: "b".to_string(),
            ty: Elem::UInt,
            vectorization: None,
        }),
        ty: Elem::UInt,
        vectorization: None,
    }))]);

    assert_eq!(expanded, expected);
}

#[test]
pub fn comptime_struct_param() {
    #[allow(unused)]
    #[cube2]
    fn struct_param(#[comptime] arg: Param) -> u32 {
        arg.a * arg.b
    }

    let expanded = struct_param::expand(Param { a: 2, b: 3 });
    let expected = Block::<u32>::new(vec![Statement::Return(lit(6u32))]);

    assert_eq!(expanded, expected);
}

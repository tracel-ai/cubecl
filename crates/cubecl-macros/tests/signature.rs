// #![allow(clippy::all)]

// use cubecl_core as cubecl;
// use cubecl_core::{
//     ir::Elem,
//     new_ir::{Expr, Expression, Operator, Variable},
//     prelude::*,
// };
// use pretty_assertions::assert_eq;
// use Elem::UInt;

// mod common;
// use common::*;

// #[test]
// pub fn const_param() {
//     #[allow(unused)]
//     #[cube]
//     fn const_param(a: u32, #[comptime] b: u32) {
//         a * b;
//     }

//     // Should fail (compile tests not working for me rn).
//     // let block = const_param::expand(
//     //     Variable::<u32> {
//     //         name: "a",
//     //         _type: PhantomData,
//     //     },
//     //     Variable::<u32> {
//     //         name: "b",
//     //         _type: PhantomData,
//     //     },
//     // );

//     let expanded =
//         const_param::expand(Variable::<u32>::new("a", false, None), 2).expression_untyped();

//     let expected = block_expr(
//         vec![expr(Expression::Binary {
//             left: var_expr("a", false, UInt),
//             operator: Operator::Mul,
//             right: Box::new(lit(2u32)),
//             ty: UInt,
//             vectorization: None,
//         })],
//         None,
//     );

//     assert_eq!(expanded, expected);
// }

// #[test]
// pub fn const_generic() {
//     #[allow(unused)]
//     #[cube]
//     fn const_generic<const D: u32>(a: u32, #[comptime] b: u32) {
//         a * b + D;
//     }

//     let expanded =
//         const_generic::expand::<3>(Variable::<u32>::new("a", false, None), 2).expression_untyped();

//     let expected = block_expr(
//         vec![expr(Expression::Binary {
//             left: Box::new(Expression::Binary {
//                 left: var_expr("a", false, UInt),
//                 operator: Operator::Mul,
//                 right: Box::new(lit(2u32)),
//                 ty: UInt,
//                 vectorization: None,
//             }),
//             operator: Operator::Add,
//             right: Box::new(lit(3u32)),
//             ty: Elem::UInt,
//             vectorization: None,
//         })],
//         None,
//     );

//     assert_eq!(expanded, expected);
// }

// #[derive(Expand)]
// struct Param {
//     a: u32,
//     b: u32,
// }

// #[test]
// pub fn struct_param() {
//     #[allow(unused)]
//     #[cube]
//     fn struct_param(arg: &Param) -> u32 {
//         arg.a * arg.b
//     }

//     let expanded = struct_param::expand(Variable::new("param", false, None)).expression_untyped();
//     let expected = block_expr(
//         vec![],
//         Some(Expression::Binary {
//             left: Box::new(Expression::FieldAccess {
//                 base: var_expr("param", false, Elem::Unit),
//                 name: "a".to_string(),
//                 ty: Elem::UInt,
//                 vectorization: None,
//             }),
//             operator: Operator::Mul,
//             right: Box::new(Expression::FieldAccess {
//                 base: var_expr("param", false, Elem::Unit),
//                 name: "b".to_string(),
//                 ty: Elem::UInt,
//                 vectorization: None,
//             }),
//             ty: Elem::UInt,
//             vectorization: None,
//         }),
//     );

//     assert_eq!(expanded, expected);
// }

// #[test]
// pub fn comptime_struct_param() {
//     #[allow(unused)]
//     #[cube]
//     fn struct_param(#[comptime] arg: Param) -> u32 {
//         arg.a * arg.b
//     }

//     let expanded = struct_param::expand(Param { a: 2, b: 3 }).expression_untyped();
//     let expected = block_expr(vec![], Some(lit(6u32)));

//     assert_eq!(expanded, expected);
// }

// #[test]
// pub fn destructure() {
//     #[allow(unused)]
//     #[cube]
//     fn destructure(arg: &Param) -> u32 {
//         let Param { a, b } = arg;
//         a * b
//     }

//     let expanded = destructure::expand(Variable::new("arg", false, None)).expression_untyped();
//     let expected = block_expr(
//         vec![
//             local_init(
//                 "a",
//                 Expression::FieldAccess {
//                     base: var_expr("arg", false, Elem::Unit),
//                     name: "a".to_string(),
//                     vectorization: None,
//                     ty: Elem::UInt,
//                 },
//                 false,
//                 None,
//             ),
//             local_init(
//                 "b",
//                 Expression::FieldAccess {
//                     base: var_expr("arg", false, Elem::Unit),
//                     name: "b".to_string(),
//                     vectorization: None,
//                     ty: Elem::UInt,
//                 },
//                 false,
//                 None,
//             ),
//         ],
//         Some(Expression::Binary {
//             left: var_expr("a", false, Elem::UInt),
//             operator: Operator::Mul,
//             right: var_expr("b", false, Elem::UInt),
//             vectorization: None,
//             ty: Elem::UInt,
//         }),
//     );

//     assert_eq!(expanded, expected);
// }

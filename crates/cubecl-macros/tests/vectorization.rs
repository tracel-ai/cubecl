// use std::num::NonZero;

// use cubecl_core as cubecl;
// use cubecl_core::{
//     cube,
//     ir::Elem,
//     new_ir::{Expr, Expression, Operator, Variable},
// };
// use pretty_assertions::assert_eq;

// mod common;
// use common::*;

// #[test]
// pub fn vectorization_simple() {
//     #[allow(unused)]
//     #[cube]
//     fn vectorized(a: u32, b: u32) -> u32 {
//         let c = a * b; // a = vec4(u32), b = u32, c = vec4(u32)
//         c * a // return = vec4(u32) * vec4(u32)
//     }

//     let expanded = vectorized::expand(
//         Variable::new("a", false, NonZero::new(4)),
//         Variable::new("b", false, None),
//     )
//     .expression_untyped();
//     let expected = block_expr(
//         vec![init_vec(
//             "c",
//             Expression::Binary {
//                 left: vec_var_expr("a", false, Elem::UInt, 4),
//                 operator: Operator::Mul,
//                 right: var_expr("b", false, Elem::UInt),
//                 vectorization: NonZero::new(4),
//                 ty: Elem::UInt,
//             },
//             false,
//             None,
//             4,
//         )],
//         Some(Expression::Binary {
//             left: vec_var_expr("c", false, Elem::UInt, 4),
//             operator: Operator::Mul,
//             right: vec_var_expr("a", false, Elem::UInt, 4),
//             vectorization: NonZero::new(4),
//             ty: Elem::UInt,
//         }),
//     );

//     assert_eq!(expanded, expected);
// }

use std::num::NonZero;

use common::*;
use cubecl_core::{self as cubecl, cube, prelude::Tensor2};
use cubecl_core::{
    ir::{Elem, IntKind},
    new_ir::*,
};
use pretty_assertions::assert_eq;

mod common;

#[test]
fn simple_index() {
    #[allow(unused)]
    #[cube]
    fn simple_index(tensor: &Tensor2<u32>) -> u32 {
        tensor[10]
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("tensor", Elem::UInt),
            index: Box::new(lit(10)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn array_index() {
    #[allow(unused)]
    #[cube]
    fn simple_index(tensor: &Tensor2<u32>) -> u32 {
        tensor[[2, 4]]
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("tensor", Elem::UInt),
            index: Box::new(Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(lit(2)),
                    operator: Operator::Mul,
                    right: Box::new(Expression::Tensor(TensorExpression::Stride {
                        tensor: var_expr("tensor", Elem::UInt),
                        dim: Box::new(lit(0)),
                    })),
                    vectorization: None,
                    ty: Elem::Int(IntKind::I32),
                }),
                operator: Operator::Add,
                right: Box::new(Expression::Binary {
                    left: Box::new(lit(4)),
                    operator: Operator::Mul,
                    right: Box::new(Expression::Tensor(TensorExpression::Stride {
                        tensor: var_expr("tensor", Elem::UInt),
                        dim: Box::new(lit(1)),
                    })),
                    vectorization: None,
                    ty: Elem::Int(IntKind::I32),
                }),
                vectorization: None,
                ty: Elem::Int(IntKind::I32),
            }),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn vectorization_tracing() {
    #[allow(unused)]
    #[cube]
    fn vectorized(tensor: &Tensor2<u32>, scalar: u32) -> u32 {
        let a = tensor[10]; //tensor: vec4, a: vec4
        a * scalar // scalar: vec2, a: vec4 split into 2xvec2, output: vec2
    }

    let expanded = vectorized::expand(
        Variable::new("tensor", NonZero::new(4)),
        Variable::new("scalar", NonZero::new(2)),
    )
    .expression_untyped();
    let expected = block_expr(
        vec![init_vec(
            "a",
            Expression::Tensor(TensorExpression::Index {
                tensor: vec_var_expr("tensor", Elem::UInt, 4),
                index: Box::new(lit(10)),
                vectorization: None,
            }),
            false,
            None,
            4,
        )],
        Some(Expression::Binary {
            left: vec_var_expr("a", Elem::UInt, 4),
            operator: Operator::Mul,
            right: vec_var_expr("scalar", Elem::UInt, 2),
            vectorization: NonZero::new(2),
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn simple_slice() {
    #[allow(unused)]
    #[cube]
    fn simple_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[5..8];
        b[1]
    }

    let expanded = simple_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(5)),
                    end: Some(Box::new(lit(8))),
                    inclusive: false,
                }],
                tensor: var_expr("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("b", Elem::UInt),
            index: Box::new(lit(1)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_open_start() {
    #[allow(unused)]
    #[cube]
    fn slice_open_start(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[..8];
        b[1]
    }

    let expanded = slice_open_start::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(0)),
                    end: Some(Box::new(lit(8))),
                    inclusive: false,
                }],
                tensor: var_expr("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("b", Elem::UInt),
            index: Box::new(lit(1)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_open_end() {
    #[allow(unused)]
    #[cube]
    fn slice_open_end(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[2..];
        b[1]
    }

    let expanded = slice_open_end::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(2)),
                    end: None,
                    inclusive: false,
                }],
                tensor: var_expr("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("b", Elem::UInt),
            index: Box::new(lit(1)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn multi_range_slice() {
    #[allow(unused)]
    #[cube]
    fn multi_range_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[[..2, ..3]];
        b[1]
    }

    let expanded = multi_range_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![
                    SliceRange {
                        start: Box::new(lit(0)),
                        end: Some(Box::new(lit(2))),
                        inclusive: false,
                    },
                    SliceRange {
                        start: Box::new(lit(0)),
                        end: Some(Box::new(lit(3))),
                        inclusive: false,
                    },
                ],
                tensor: var_expr("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("b", Elem::UInt),
            index: Box::new(lit(1)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_different_range_types() {
    #[allow(unused)]
    #[cube]
    fn multi_range_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[(.., 2..4)];
        b[1]
    }

    let expanded = multi_range_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![
                    SliceRange {
                        start: Box::new(lit(0)),
                        end: None,
                        inclusive: false,
                    },
                    SliceRange {
                        start: Box::new(lit(2)),
                        end: Some(Box::new(lit(4))),
                        inclusive: false,
                    },
                ],
                tensor: var_expr("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var_expr("b", Elem::UInt),
            index: Box::new(lit(1)),
            vectorization: None,
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn mut_index() {
    #[allow(unused)]
    #[cube]
    fn simple_index(tensor: &mut Tensor2<u32>) {
        tensor[10] = 1;
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block_expr(
        vec![expr(Expression::Assigment {
            left: Box::new(Expression::Tensor(TensorExpression::Index {
                tensor: var_expr("tensor", Elem::UInt),
                index: Box::new(lit(10)),
                vectorization: None,
            })),
            right: Box::new(lit(1u32)),
            vectorization: None,
            ty: Elem::UInt,
        })],
        None,
    );

    assert_eq!(expanded, expected);
}

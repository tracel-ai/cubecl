use std::num::NonZero;

use common::*;
use cubecl_core::{
    ir::{Elem, IntKind},
    new_ir::{
        element::Tensor2, Expr, Expression, Operator, SliceRange, TensorExpression, Variable,
    },
};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;

#[test]
fn simple_index() {
    #[allow(unused)]
    #[cube2]
    fn simple_index(tensor: &Tensor2<u32>) -> u32 {
        tensor[10]
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("tensor", Elem::UInt),
            index: Box::new(lit(10)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn array_index() {
    #[allow(unused)]
    #[cube2]
    fn simple_index(tensor: &Tensor2<u32>) -> u32 {
        tensor[[2, 4]]
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("tensor", Elem::UInt),
            index: Box::new(Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(lit(2)),
                    operator: Operator::Mul,
                    right: Box::new(Expression::Tensor(TensorExpression::Stride {
                        tensor: var("tensor", Elem::UInt),
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
                        tensor: var("tensor", Elem::UInt),
                        dim: Box::new(lit(1)),
                    })),
                    vectorization: None,
                    ty: Elem::Int(IntKind::I32),
                }),
                vectorization: None,
                ty: Elem::Int(IntKind::I32),
            }),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn vectorization_tracing() {
    #[allow(unused)]
    #[cube2]
    fn vectorized(tensor: &Tensor2<u32>, scalar: u32) -> u32 {
        let a = tensor[10];
        a * scalar
    }

    let expanded = vectorized::expand(
        Variable::new("tensor", NonZero::new(4)),
        Variable::new("scalar", NonZero::new(2)),
    )
    .expression_untyped();
    let expected = block(
        vec![init_vec(
            "a",
            Expression::Tensor(TensorExpression::Index {
                tensor: vec_var("tensor", Elem::UInt, 4),
                index: Box::new(lit(10)),
            }),
            false,
            None,
            4,
        )],
        Some(Expression::Binary {
            left: vec_var("a", Elem::UInt, 4),
            operator: Operator::Mul,
            right: vec_var("scalar", Elem::UInt, 2),
            vectorization: NonZero::new(2),
            ty: Elem::UInt,
        }),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn simple_slice() {
    #[allow(unused)]
    #[cube2]
    fn simple_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[5..8];
        b[1]
    }

    let expanded = simple_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(5)),
                    end: Some(Box::new(lit(8))),
                    inclusive: false,
                }],
                tensor: var("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("b", Elem::UInt),
            index: Box::new(lit(1)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_open_start() {
    #[allow(unused)]
    #[cube2]
    fn slice_open_start(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[..8];
        b[1]
    }

    let expanded = slice_open_start::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(0)),
                    end: Some(Box::new(lit(8))),
                    inclusive: false,
                }],
                tensor: var("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("b", Elem::UInt),
            index: Box::new(lit(1)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_open_end() {
    #[allow(unused)]
    #[cube2]
    fn slice_open_end(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[2..];
        b[1]
    }

    let expanded = slice_open_end::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![local_init(
            "b",
            Expression::Tensor(TensorExpression::Slice {
                ranges: vec![SliceRange {
                    start: Box::new(lit(2)),
                    end: None,
                    inclusive: false,
                }],
                tensor: var("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("b", Elem::UInt),
            index: Box::new(lit(1)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn multi_range_slice() {
    #[allow(unused)]
    #[cube2]
    fn multi_range_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[[..2, ..3]];
        b[1]
    }

    let expanded = multi_range_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
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
                tensor: var("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("b", Elem::UInt),
            index: Box::new(lit(1)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn slice_different_range_types() {
    #[allow(unused)]
    #[cube2]
    fn multi_range_slice(tensor: &Tensor2<u32>) -> u32 {
        let b = &tensor[(.., 2..4)];
        b[1]
    }

    let expanded = multi_range_slice::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
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
                tensor: var("tensor", Elem::UInt),
            }),
            false,
            None,
        )],
        Some(Expression::Tensor(TensorExpression::Index {
            tensor: var("b", Elem::UInt),
            index: Box::new(lit(1)),
        })),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn mut_index() {
    #[allow(unused)]
    #[cube2]
    fn simple_index(tensor: &mut Tensor2<u32>) {
        tensor[10] = 1;
    }

    let expanded = simple_index::expand(Variable::new("tensor", None)).expression_untyped();
    let expected = block(
        vec![expr(Expression::Assigment {
            left: Box::new(Expression::Tensor(TensorExpression::Index {
                tensor: var("tensor", Elem::UInt),
                index: Box::new(lit(10)),
            })),
            right: Box::new(lit(1u32)),
            vectorization: None,
            ty: Elem::UInt,
        })],
        None,
    );

    assert_eq!(expanded, expected);
}

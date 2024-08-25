use cubecl_core::{
    ir::Elem,
    new_ir::{Expr, Expression, Operator, Range, Statement, Variable},
};
use cubecl_macros_2::cube2;
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[test]
fn for_loop() {
    #[allow(unused)]
    #[cube2]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: Box::new(lit(2u32)),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![Statement::Expression(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_inclusive() {
    #[allow(unused)]
    #[cube2]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in 0..=2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: Box::new(lit(2u32)),
                    step: None,
                    inclusive: true,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![Statement::Expression(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_stepped() {
    #[allow(unused)]
    #[cube2]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in (0..2).step_by(3) {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: Box::new(lit(2u32)),
                    step: Some(Box::new(lit(3u32))),
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![Statement::Expression(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll() {
    #[allow(unused)]
    #[cube2]
    fn for_loop() -> u32 {
        let mut a = 0;
        #[unroll]
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: Box::new(lit(2u32)),
                    step: None,
                    inclusive: false,
                },
                unroll: true,
                variable: var("i", Elem::UInt),
                block: vec![expr(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll_comptime() {
    #[allow(unused)]
    #[cube2]
    fn for_loop(#[comptime] should_unroll: bool) -> u32 {
        let mut a = 0;
        #[unroll(should_unroll)]
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand(false).expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: Box::new(lit(2u32)),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![expr(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
#[should_panic(expected = "Can't unroll loop with dynamic end")]
fn for_loop_unroll_dynamic_fails() {
    #[allow(unused)]
    #[cube2]
    fn for_loop(loop_end: u32) -> u32 {
        let mut a = 0;
        #[unroll]
        for i in 0..loop_end {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand(Variable::new("end", None)).expression_untyped();
    let expected = block(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: var("end", Elem::UInt),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![expr(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll_comptime_bounds() {
    #[allow(unused)]
    #[cube2]
    fn for_loop(dyn_end: u32, #[comptime] end: Option<u32>) -> u32 {
        let should_unroll = end.is_some();
        let end = end.unwrap_or(dyn_end);
        let mut a = 0;
        #[unroll(should_unroll)]
        for i in 0..end {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand(Variable::new("a", None), None).expression_untyped();
    let expected = block(
        vec![
            local_init("end", *var("a", Elem::UInt), false, None),
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: var("end", Elem::UInt),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: vec![expr(Expression::Binary {
                    left: var("a", Elem::UInt),
                    operator: Operator::AddAssign,
                    right: var("i", Elem::UInt),
                    vectorization: None,
                    ty: Elem::UInt,
                })],
            }),
        ],
        Some(*var("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

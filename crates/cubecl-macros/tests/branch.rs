#![allow(clippy::all)]

use cubecl_core::{ir::Elem, new_ir::*, prelude::*};
use pretty_assertions::assert_eq;

mod common;
use common::*;

#[test]
fn for_loop() {
    #[allow(unused)]
    #[cube]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block_expr(
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
                block: block(
                    vec![Statement::Expression(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_inclusive() {
    #[allow(unused)]
    #[cube]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in 0..=2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block_expr(
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
                block: block(
                    vec![Statement::Expression(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_stepped() {
    #[allow(unused)]
    #[cube]
    fn for_loop() -> u32 {
        let mut a = 0;
        for i in (0..2).step_by(3) {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block_expr(
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
                block: block(
                    vec![Statement::Expression(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll() {
    #[allow(unused)]
    #[cube]
    fn for_loop() -> u32 {
        let mut a = 0;
        #[unroll]
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand().expression_untyped();
    let expected = block_expr(
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
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll_comptime() {
    #[allow(unused)]
    #[cube]
    fn for_loop(#[comptime] should_unroll: bool) -> u32 {
        let mut a = 0;
        #[unroll(should_unroll)]
        for i in 0..2 {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand(false).expression_untyped();
    let expected = block_expr(
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
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
#[should_panic(expected = "Can't unroll loop with dynamic end")]
fn for_loop_unroll_dynamic_fails() {
    #[allow(unused)]
    #[cube]
    fn for_loop(loop_end: u32) -> u32 {
        let mut a = 0;
        #[unroll]
        for i in 0..loop_end {
            a += i;
        }
        a
    }

    let expanded = for_loop::expand(Variable::new("end", None)).expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: var_expr("end", Elem::UInt),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn for_loop_unroll_comptime_bounds() {
    #[allow(unused)]
    #[cube]
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
    let expected = block_expr(
        vec![
            local_init("end", *var_expr("a", Elem::UInt), false, None),
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::ForLoop {
                range: Range {
                    start: Box::new(lit(0u32)),
                    end: var_expr("end", Elem::UInt),
                    step: None,
                    inclusive: false,
                },
                unroll: false,
                variable: var("i", Elem::UInt),
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: var_expr("i", Elem::UInt),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn while_loop() {
    #[allow(unused)]
    #[cube]
    fn while_loop() -> u32 {
        let mut a = 0;
        while a % 4 != 0 {
            a += 1;
        }
        a
    }

    let expanded = while_loop::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::WhileLoop {
                condition: Box::new(Expression::Binary {
                    left: Box::new(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::Rem,
                        right: Box::new(lit(4u32)),
                        vectorization: None,
                        ty: Elem::UInt,
                    }),
                    operator: Operator::Ne,
                    right: Box::new(lit(0u32)),
                    vectorization: None,
                    ty: Elem::Bool,
                }),
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: Box::new(lit(1u32)),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn loop_expr() {
    #[allow(unused)]
    #[cube]
    fn loop_expr() -> u32 {
        let mut a = 0;
        loop {
            a += 1;
        }
        a
    }

    let expanded = loop_expr::expand().expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::Loop {
                block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: Box::new(lit(1u32)),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn if_expr() {
    #[allow(unused)]
    #[cube]
    fn if_expr(cond: bool) -> u32 {
        let mut a = 0;
        if cond {
            a += 1;
        } else {
            a += 2;
        }
        a
    }

    let expanded = if_expr::expand(Variable::new("cond", None)).expression_untyped();
    let expected = block_expr(
        vec![
            local_init("a", lit(0u32), true, None),
            Statement::Expression(Expression::If {
                condition: var_expr("cond", Elem::Bool),
                then_block: block(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: Box::new(lit(1u32)),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ),
                else_branch: Some(Box::new(block_expr(
                    vec![expr(Expression::Binary {
                        left: var_expr("a", Elem::UInt),
                        operator: Operator::AddAssign,
                        right: Box::new(lit(2u32)),
                        vectorization: None,
                        ty: Elem::UInt,
                    })],
                    None,
                ))),
            }),
        ],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn if_returns() {
    #[allow(unused)]
    #[cube]
    fn if_returns(cond: bool) -> u32 {
        let a = if cond { 1 } else { 2 };
        a
    }

    let expanded = if_returns::expand(Variable::new("cond", None)).expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "a",
            Expression::If {
                condition: var_expr("cond", Elem::Bool),
                then_block: block(vec![], Some(lit(1u32))),
                else_branch: Some(Box::new(block_expr(vec![], Some(lit(2u32))))),
            },
            false,
            None,
        )],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn chained_if() {
    #[allow(unused)]
    #[cube]
    fn if_returns(cond1: bool, cond2: bool) -> u32 {
        let a = if cond1 {
            1
        } else if cond2 {
            2
        } else {
            3
        };
        a
    }

    let expanded = if_returns::expand(Variable::new("cond1", None), Variable::new("cond2", None))
        .expression_untyped();
    let expected = block_expr(
        vec![local_init(
            "a",
            Expression::If {
                condition: var_expr("cond1", Elem::Bool),
                then_block: block(vec![], Some(lit(1u32))),
                else_branch: Some(Box::new(Expression::If {
                    condition: var_expr("cond2", Elem::Bool),
                    then_block: block(vec![], Some(lit(2u32))),
                    else_branch: Some(Box::new(block_expr(vec![], Some(lit(3u32))))),
                })),
            },
            false,
            None,
        )],
        Some(*var_expr("a", Elem::UInt)),
    );

    assert_eq!(expanded, expected);
}

#[test]
fn explicit_return() {
    #[allow(unused)]
    #[cube]
    fn if_returns(cond: bool) -> u32 {
        if cond {
            return 10;
        }
        1
    }

    let expanded = if_returns::expand(Variable::new("cond", None)).expression_untyped();
    let expected = block_expr(
        vec![expr(Expression::If {
            condition: var_expr("cond", Elem::Bool),
            then_block: block(
                vec![expr(Expression::Return {
                    expr: Some(Box::new(lit(10u32))),
                })],
                None,
            ),
            else_branch: None,
        })],
        Some(lit(1u32)),
    );

    assert_eq!(expanded, expected);
}

//! Regression tests for remainder folding: `x % 1` is zero, never `x`.
//!
//! A `simplify!` fold used to rewrite `x % 1` to its `lhs`. After
//! `QuotientRange` emits `exact = (scaled % max(s, 1)) == 0`, a constant
//! `s = 1` folds the divisor to one and the bad fold turned the check into
//! `scaled == 0`, dropping every tap whose numerator was nonzero. The fold
//! belongs to `const_eval!` (SCCP), which can materialize a zero constant;
//! `SimplifyInterface::check_fold` can only forward existing values.

use cubecl_ir::rewrite::SimplifyOps;
use cubecl_opt::passes::sccp::sccp;
use expect_test::expect;
use pliron::{
    context::Context,
    init_env_logger_for_tests,
    irbuild::{
        IRStatus,
        match_rewrite::{RewriterOrder, apply_match_rewrite},
    },
    irfmt::parsers::spaced,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    result::{ExpectOk, Result},
};

fn run_simplify(input: &str) -> Result<(IRStatus, String)> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(op, ctx)?;

    let mut pass = SimplifyOps;
    let changed = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    verify_operation(op, ctx)?;

    Ok((
        changed,
        Operation::get_op_dyn(op, ctx).disp(ctx).to_string(),
    ))
}

/// `math.u_rem (x, 1)` over `cube.index`, the shape `QuotientRange` emits.
fn u_rem_by_one() -> String {
    r#"
    builtin.func @f: builtin.function <(cube.index) -> (cube.index)> [] {
      ^entry(x: cube.index):
        one = builtin.constant <cube.index 1> : cube.index;
        rem = math.u_rem (x, one) [] []: <(cube.index , cube.index ) -> (cube.index )>;
        branch.return rem
    }
  "#
    .to_string()
}

/// `math.s_rem (x, 1)` over a signed integer, the same bug in `SRemOp`.
fn s_rem_by_one() -> String {
    r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
        one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        rem = math.s_rem (x, one) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return rem
    }
  "#
    .to_string()
}

/// Simplify must leave `x % 1` alone rather than rewrite it to `x`.
#[test]
fn simplify_does_not_rewrite_rem_by_one_to_its_lhs() -> Result<()> {
    for input in [u_rem_by_one(), s_rem_by_one()] {
        let (changed, _) = run_simplify(&input)?;
        assert_eq!(
            changed,
            IRStatus::Unchanged,
            "x % 1 must not fold to x under SimplifyOps"
        );
    }
    Ok(())
}

/// SCCP folds `x % 1` to a zero constant through `const_eval!`.
#[test]
fn sccp_folds_rem_by_one_to_zero() -> Result<()> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op =
        parse_from_str(spaced(Operation::top_level_parser()), ctx, &u_rem_by_one()).expect_ok(ctx);
    verify_operation(op, ctx)?;

    sccp(op, ctx)?;
    let after = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();
    expect![[r#"
        builtin.func @f: builtin.function <(cube.index ) -> (cube.index )> 
        {
          ^entry_block1v1(x_v0: cube.index ) !0:
            one_v1 = builtin.constant <cube.index 1> : cube.index  !1;
            rem_v3 = builtin.constant <cube.index 0> : cube.index  !2;
            rem_v2 = math.u_rem (x_v0, one_v1) [] []: <(cube.index , cube.index ) -> (cube.index )> !3;
            branch.return rem_v3 !4
        }"#]].assert_eq(&after);
    Ok(())
}

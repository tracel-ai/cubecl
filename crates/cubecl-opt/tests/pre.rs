use cubecl_ir::prelude::*;
use cubecl_opt::{
    analyses::dataflow_solver::pre::{AnticipatedMode, InsertMode},
    passes::pre::run_pre,
};
use expect_test::expect;
use pliron::{
    init_env_logger_for_tests, irfmt::parsers::spaced, operation::verify_operation,
    parsable::parse_from_str, printable::Printable, result::ExpectOk,
};

fn run_pre_on_text(input: &str, antic: AnticipatedMode, insert: InsertMode) -> Result<String> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    run_pre(ctx, op, antic, insert)?;

    Ok(op.disp(ctx).to_string())
}

fn run_lcm_on_text(input: &str) -> Result<String> {
    run_pre_on_text(input, AnticipatedMode::Classical, InsertMode::Lazy)
}

fn run_scm_on_text(input: &str) -> Result<String> {
    run_pre_on_text(input, AnticipatedMode::Classical, InsertMode::Sparse)
}

fn run_sscm_on_text(input: &str) -> Result<String> {
    run_pre_on_text(input, AnticipatedMode::Speculative, InsertMode::Sparse)
}

// Example from the Branch Divergence Reduction Based on Code Motion paper to validate the different
// insertion strategies
fn sscm_paper_example() -> &'static str {
    r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(
          builtin.integer i64, builtin.integer i64, cube.bool, cube.index
      ) -> (builtin.integer i64)>
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(a: builtin.integer i64, b: builtin.integer i64, outer_cond: cube.bool, lo: cube.index):
          result = scf.if outer_cond : builtin.integer i64 then {
            ^outer_then():
              stride = builtin.constant <cube.index 1> : cube.index;
              hi = cube.read_builtin (UnitPos) : cube.index;

              scf.for lo to hi step stride iter_args() {
                ^loop_body(iv: builtin.integer i64):
                  x = math.i_add (a, b) [] []:
                      <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
                  branch.yield ()
              };
              branch.yield (a)
          } else {
            ^outer_else():
              inner_cond = plane.elect () [] []: <() -> (cube.bool)>;

              inner = scf.if inner_cond : builtin.integer i64 then {
                ^inner_then():
                  y = math.i_add (a, b) [] []:
                      <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
                  branch.yield (y)
              } else {
                ^inner_else():
                  branch.yield (a)
              };

              z = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield (z)
          };

          branch.return result
      }
    }
    "#
}

#[test]
fn insert_paper_example_lcm() -> Result<()> {
    let input = sscm_paper_example();
    let actual = run_lcm_on_text(input)?;

    expect![[r#""#]].assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_scm() -> Result<()> {
    let input = sscm_paper_example();
    let actual = run_scm_on_text(input)?;

    expect![[r#""#]].assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_sscm() -> Result<()> {
    let input = sscm_paper_example();
    let actual = run_sscm_on_text(input)?;

    expect![[r#""#]].assert_eq(&actual);

    Ok(())
}

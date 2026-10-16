use cubecl_ir::{HasSideEffects, prelude::*};
use cubecl_opt::{
    analyses::dataflow_solver::pre::{AnticipatedMode, InsertMode},
    passes::pre::run_pre,
};
use expect_test::expect;
use pliron::{
    init_env_logger_for_tests, irbuild::listener::DummyListener, irfmt::parsers::spaced,
    operation::verify_operation, parsable::parse_from_str, result::ExpectOk,
};

#[pliron_op(name = "pre.black_box", format = "$0", verifier = "succ")]
#[op_traits(HasSideEffects)]
pub struct BlackBox;

fn run_pre_on_text(
    input: &str,
    antic: AnticipatedMode,
    insert: InsertMode,
) -> Result<(IRStatus, String)> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let mut rewriter = IRRewriter::<DummyListener>::default();
    let status = run_pre(ctx, op, &mut rewriter, antic, insert)?;
    let after = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();

    Ok((status, after))
}

fn run_lcm_on_text(input: &str) -> Result<(IRStatus, String)> {
    run_pre_on_text(input, AnticipatedMode::Classical, InsertMode::Lazy)
}

fn run_scm_on_text(input: &str) -> Result<(IRStatus, String)> {
    run_pre_on_text(input, AnticipatedMode::Classical, InsertMode::Sparse)
}

fn run_sscm_on_text(input: &str) -> Result<(IRStatus, String)> {
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
                  uses_x = math.i_add (x, x) [] []:
                    <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
                  pre.black_box uses_x;
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
    let (status, actual) = run_lcm_on_text(input)?;

    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.module @module 
        {
          ^module_block_block1v1() !0:
            builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, cube.bool , cube.index ) -> (builtin.integer i64)> 
              [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]
            {
              ^entry_block2v1(a_v0: builtin.integer i64, b_v1: builtin.integer i64, outer_cond_v2: cube.bool , lo_v3: cube.index ) !1:
                result_v13 = scf.if outer_cond_v2 : builtin.integer i64 then 
                {
                  ^outer_then_block3v1() !2:
                    stride_v4 = builtin.constant <cube.index 1> : cube.index  !3;
                    hi_v5 = cube.read_builtin (UnitPos) : cube.index  !4;
                    scf.for lo_v3 to hi_v5 step stride_v4 
                    {
                      ^loop_body_block4v1(iv_v6: builtin.integer i64) !5:
                        x_v7 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !6;
                        uses_x_v8 = math.i_add (x_v7, x_v7) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !7;
                        pre.black_box uses_x_v8 !8;
                        branch.yield () !9
                    } !10;
                    branch.yield (a_v0) !11
                } else 
                {
                  ^outer_else_block5v1() !12:
                    inner_cond_v9 = plane.elect () [] []: <() -> (cube.bool )> !13;
                    inner_v11, z_v15 = scf.if inner_cond_v9 : builtin.integer i64, builtin.integer i64 then 
                    {
                      ^inner_then_block6v1() !14:
                        y_v10 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !15;
                        branch.yield (y_v10, y_v10) !16
                    } else 
                    {
                      ^inner_else_block7v1() !17:
                        z_v14 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !18;
                        branch.yield (a_v0, z_v14) !19
                    } !20;
                    branch.yield (z_v15) !21
                } !22;
                branch.return result_v13 !23
            } !24
        }"#]].assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_scm() -> Result<()> {
    let input = sscm_paper_example();
    let (status, actual) = run_scm_on_text(input)?;

    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.module @module 
        {
          ^module_block_block1v1() !0:
            builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, cube.bool , cube.index ) -> (builtin.integer i64)> 
              [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]
            {
              ^entry_block2v1(a_v0: builtin.integer i64, b_v1: builtin.integer i64, outer_cond_v2: cube.bool , lo_v3: cube.index ) !1:
                result_v13 = scf.if outer_cond_v2 : builtin.integer i64 then 
                {
                  ^outer_then_block3v1() !2:
                    stride_v4 = builtin.constant <cube.index 1> : cube.index  !3;
                    hi_v5 = cube.read_builtin (UnitPos) : cube.index  !4;
                    scf.for lo_v3 to hi_v5 step stride_v4 
                    {
                      ^loop_body_block4v1(iv_v6: builtin.integer i64) !5:
                        x_v7 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !6;
                        uses_x_v8 = math.i_add (x_v7, x_v7) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !7;
                        pre.black_box uses_x_v8 !8;
                        branch.yield () !9
                    } !10;
                    branch.yield (a_v0) !11
                } else 
                {
                  ^outer_else_block5v1() !12:
                    z_v14 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !13;
                    branch.yield (z_v14) !14
                } !15;
                branch.return result_v13 !16
            } !17
        }"#]].assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_sscm() -> Result<()> {
    let input = sscm_paper_example();
    let (status, actual) = run_sscm_on_text(input)?;

    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.module @module 
        {
          ^module_block_block1v1() !0:
            builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, cube.bool , cube.index ) -> (builtin.integer i64)> 
              [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]
            {
              ^entry_block2v1(a_v0: builtin.integer i64, b_v1: builtin.integer i64, outer_cond_v2: cube.bool , lo_v3: cube.index ) !1:
                z_v14 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !2;
                result_v13 = scf.if outer_cond_v2 : builtin.integer i64 then 
                {
                  ^outer_then_block3v1() !3:
                    stride_v4 = builtin.constant <cube.index 1> : cube.index  !4;
                    hi_v5 = cube.read_builtin (UnitPos) : cube.index  !5;
                    scf.for lo_v3 to hi_v5 step stride_v4 
                    {
                      ^loop_body_block4v1(iv_v6: builtin.integer i64) !6:
                        uses_x_v8 = math.i_add (z_v14, z_v14) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !7;
                        pre.black_box uses_x_v8 !8;
                        branch.yield () !9
                    } !10;
                    branch.yield (a_v0) !11
                } else 
                {
                  ^outer_else_block5v1() !12:
                    branch.yield (z_v14) !13
                } !14;
                branch.return result_v13 !15
            } !16
        }"#]].assert_eq(&actual);

    Ok(())
}

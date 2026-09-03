// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron contributors

//! SCCP integration tests using textual LLVM dialect IR parsing.

use cubecl_opt::passes::sccp::sccp;
use expect_test::expect;
use pliron::{
    context::Context,
    init_env_logger_for_tests,
    irbuild::IRStatus,
    irfmt::parsers::spaced,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    result::{ExpectOk, Result},
};

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    derive::pliron_op,
};

#[pliron_op(
    name = "test.test_region",
    format = "region($0)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ"
)]
pub struct TestRegionOp;

#[pliron_op(
    name = "test.test_two_regions",
    format = "region($0) ` ` region($1)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    verifier = "succ"
)]
pub struct TestTwoRegionsOp;

fn run_sccp_on_text(input: &str) -> Result<(IRStatus, String)> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let status = sccp(op, ctx)?;

    let after = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();
    log::trace!("After SCCP:\n{}", after);
    verify_operation(op, ctx)?;
    Ok((status, after))
}

#[test]
fn sccp_folds_add_of_two_constants() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      a = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      b = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
      sum = math.i_add (a, b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return sum
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            a_v0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !1;
            b_v1 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !2;
            sum_v3 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !3;
            sum_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !4;
            branch.return sum_v3 !5
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_is_path_sensitive() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
      y = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
      one = builtin.constant <cube.bool true> : cube.bool;
      cf.branch_conditional if one ^bb0(x, y) else ^bb1(x, y)

      ^bb0(x0: builtin.integer i64,y0: builtin.integer i64):
      cf.branch ^bb2(y0, y0)

      ^bb1(x1: builtin.integer i64,y1: builtin.integer i64):
      cf.branch ^bb2(x1, y1)

      ^bb2(x2: builtin.integer i64,y2: builtin.integer i64):
      z = math.i_add (x2, y2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return z
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(x_v0: builtin.integer i64) !0:
            y_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !1;
            one_v2 = builtin.constant <cube.bool true> : cube.bool  !2;
            cf.branch_conditional if one_v2 ^bb0_block4v1(x_v0, y_v1) else ^bb1_block5v1(x_v0, y_v1) !3

          ^bb0_block4v1(x0_v3: builtin.integer i64, y0_v4: builtin.integer i64) !4:
            y0_v10 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !5;
            cf.branch ^bb2_block3v3(y0_v10, y0_v10) !6

          ^bb1_block5v1(x1_v5: builtin.integer i64, y1_v6: builtin.integer i64) !7:
            cf.branch ^bb2_block3v3(x1_v5, y1_v6) !8

          ^bb2_block3v3(x2_v7: builtin.integer i64, y2_v8: builtin.integer i64) !9:
            x2_v12 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !10;
            y2_v13 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !11;
            z_v11 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !12;
            z_v9 = math.i_add (x2_v12, y2_v13) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !13;
            branch.return z_v11 !14
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_folded_condition_makes_branch_dead() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
      y = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
      zero_i1 = builtin.constant <cube.bool false> : cube.bool;
      one_i1 = builtin.constant <cube.bool true> : cube.bool;
      one = cube.bool_or (zero_i1, one_i1) [] []: <(cube.bool, cube.bool) -> (cube.bool)>;
      cf.branch_conditional if one ^bb0(x, y) else ^bb1(x, y)

      ^bb0(x0: builtin.integer i64,y0: builtin.integer i64):
      cf.branch ^bb2(y0, y0)

      ^bb1(x1: builtin.integer i64,y1: builtin.integer i64):
      cf.branch ^bb2(x1, y1)

      ^bb2(x2: builtin.integer i64,y2: builtin.integer i64):
      z = math.i_add (x2, y2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return z
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(x_v0: builtin.integer i64) !0:
            y_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !1;
            zero_i1_v2 = builtin.constant <cube.bool false> : cube.bool  !2;
            one_i1_v3 = builtin.constant <cube.bool true> : cube.bool  !3;
            one_v12 = builtin.constant <cube.bool true> : cube.bool  !4;
            one_v4 = cube.bool_or (zero_i1_v2, one_i1_v3) [] []: <(cube.bool , cube.bool ) -> (cube.bool )> !5;
            cf.branch_conditional if one_v12 ^bb0_block4v1(x_v0, y_v1) else ^bb1_block5v1(x_v0, y_v1) !6

          ^bb0_block4v1(x0_v5: builtin.integer i64, y0_v6: builtin.integer i64) !7:
            y0_v13 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !8;
            cf.branch ^bb2_block3v3(y0_v13, y0_v13) !9

          ^bb1_block5v1(x1_v7: builtin.integer i64, y1_v8: builtin.integer i64) !10:
            cf.branch ^bb2_block3v3(x1_v7, y1_v8) !11

          ^bb2_block3v3(x2_v9: builtin.integer i64, y2_v10: builtin.integer i64) !12:
            x2_v15 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !13;
            y2_v16 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !14;
            z_v14 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !15;
            z_v11 = math.i_add (x2_v15, y2_v16) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !16;
            branch.return z_v14 !17
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_meets_distinct_constants_from_live_predecessors_as_not_a_constant() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
      a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      b0 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
      cf.branch ^bb2(a0, b0)

      ^bb1():
      a1 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      b1 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
      cf.branch ^bb2(a1, b1)

      ^bb2(x: builtin.integer i64, y: builtin.integer i64):
      x_plus_y = math.i_add (x, y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      y_plus_y = math.i_add (y, y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      result = math.i_add (x_plus_y, y_plus_y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return result
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1() !1

          ^bb0_block4v1() !2:
            a0_v1 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !3;
            b0_v2 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !4;
            cf.branch ^bb2_block3v3(a0_v1, b0_v2) !5

          ^bb1_block5v1() !6:
            a1_v3 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !7;
            b1_v4 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !8;
            cf.branch ^bb2_block3v3(a1_v3, b1_v4) !9

          ^bb2_block3v3(x_v5: builtin.integer i64, y_v6: builtin.integer i64) !10:
            y_v11 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !11;
            x_plus_y_v7 = math.i_add (x_v5, y_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !12;
            y_plus_y_v10 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64 !13;
            y_plus_y_v8 = math.i_add (y_v11, y_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !14;
            result_v9 = math.i_add (x_plus_y_v7, y_plus_y_v10) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !15;
            branch.return result_v9 !16
        }"#]].assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_is_path_sensitive_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
        y = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        one = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        one_b = cmp.i_equal (one, one) [] []: <(builtin.integer i32, builtin.integer i32) -> (cube.bool )>;
        x2, y2 = scf.if one_b : builtin.integer i64, builtin.integer i64 then {
          ^then_block():
            branch.yield (y, y)
        } else {
          ^else_block():
            branch.yield (x, y)
        };
        z = math.i_add (x2, y2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return z
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(x_v0: builtin.integer i64) !0:
            y_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !1;
            one_v2 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32 !2;
            one_b_v10 = builtin.constant <cube.bool true> : cube.bool  !3;
            one_b_v3 = cmp.i_equal (one_v2, one_v2) [] []: <(builtin.integer i32, builtin.integer i32) -> (cube.bool )> !4;
            x2_v4, y2_v5 = scf.if one_b_v10 : builtin.integer i64, builtin.integer i64 then 
            {
              ^then_block_block2v1() !5:
                branch.yield (y_v1, y_v1) !6
            } else 
            {
              ^else_block_block3v1() !7:
                branch.yield (x_v0, y_v1) !8
            } !9;
            x2_v8 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !10;
            y2_v9 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !11;
            z_v7 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !12;
            z_v6 = math.i_add (x2_v8, y2_v9) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !13;
            branch.return z_v7 !14
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_folded_condition_makes_branch_dead_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
        y = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
        zero_i1 = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32;
        one_i1 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
        one = math.i_add (zero_i1, one_i1) [] []: <(builtin.integer i32, builtin.integer i32) -> (builtin.integer i32)>;
        one_b = cmp.i_equal (one, one_i1) [] []: <(builtin.integer i32, builtin.integer i32) -> (cube.bool )>;
        x2, y2 = scf.if one_b : builtin.integer i64, builtin.integer i64 then {
          ^then():
            branch.yield (y, y)
        } else {
          ^else():
            branch.yield (x, y)
        };
        z = math.i_add (x2, y2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return z
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(x_v0: builtin.integer i64) !0:
            y_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !1;
            zero_i1_v2 = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32 !2;
            one_i1_v3 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32 !3;
            one_v13 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32 !4;
            one_v4 = math.i_add (zero_i1_v2, one_i1_v3) [] []: <(builtin.integer i32, builtin.integer i32) -> (builtin.integer i32)> !5;
            one_b_v12 = builtin.constant <cube.bool true> : cube.bool  !6;
            one_b_v5 = cmp.i_equal (one_v13, one_i1_v3) [] []: <(builtin.integer i32, builtin.integer i32) -> (cube.bool )> !7;
            x2_v6, y2_v7 = scf.if one_b_v12 : builtin.integer i64, builtin.integer i64 then 
            {
              ^then_block2v1() !8:
                branch.yield (y_v1, y_v1) !9
            } else 
            {
              ^else_block3v1() !10:
                branch.yield (x_v0, y_v1) !11
            } !12;
            x2_v10 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !13;
            y2_v11 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !14;
            z_v9 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !15;
            z_v8 = math.i_add (x2_v10, y2_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !16;
            branch.return z_v9 !17
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_meets_distinct_constants_from_live_predecessors_as_not_a_constant_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
        x, y = scf.if cond : builtin.integer i64, builtin.integer i64 then {
          ^then_block():
            a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
            b0 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
            branch.yield (a0, b0)
        } else {
          ^else_block():
            a1 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
            b1 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
            branch.yield (a1, b1)
        };
        x_plus_y = math.i_add (x, y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        y_plus_y = math.i_add (y, y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        result = math.i_add (x_plus_y, y_plus_y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return result
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            x_v5, y_v6 = scf.if cond_v0 : builtin.integer i64, builtin.integer i64 then 
            {
              ^then_block_block2v1() !1:
                a0_v1 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !2;
                b0_v2 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !3;
                branch.yield (a0_v1, b0_v2) !4
            } else 
            {
              ^else_block_block3v1() !5:
                a1_v3 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !6;
                b1_v4 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !7;
                branch.yield (a1_v3, b1_v4) !8
            } !9;
            y_v11 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !10;
            x_plus_y_v7 = math.i_add (x_v5, y_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !11;
            y_plus_y_v10 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64 !12;
            y_plus_y_v8 = math.i_add (y_v11, y_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !13;
            result_v9 = math.i_add (x_plus_y_v7, y_plus_y_v10) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !14;
            branch.return result_v9 !15
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_is_path_sensitive_2() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> [] {
      ^entry(x: builtin.integer i64):
      y = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
      one = builtin.constant <cube.bool true> : cube.bool;
      cf.branch_conditional if one ^bb1(x, y) else ^bb0(x, y)

      ^bb0(x0: builtin.integer i64,y0: builtin.integer i64):
      cf.branch ^bb2(y0, y0)

      ^bb1(x1: builtin.integer i64,y1: builtin.integer i64):
      cf.branch ^bb2(x1, y1)

      ^bb2(x2: builtin.integer i64,y2: builtin.integer i64):
      z = math.i_add (x2, y2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return z
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    // Materialized constants inserted into ^bb0, ^bb1, and ^bb2
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(x_v0: builtin.integer i64) !0:
            y_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !1;
            one_v2 = builtin.constant <cube.bool true> : cube.bool  !2;
            cf.branch_conditional if one_v2 ^bb1_block5v1(x_v0, y_v1) else ^bb0_block4v1(x_v0, y_v1) !3

          ^bb0_block4v1(x0_v3: builtin.integer i64, y0_v4: builtin.integer i64) !4:
            cf.branch ^bb2_block2v3(y0_v4, y0_v4) !5

          ^bb1_block5v1(x1_v5: builtin.integer i64, y1_v6: builtin.integer i64) !6:
            y1_v10 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !7;
            cf.branch ^bb2_block2v3(x1_v5, y1_v10) !8

          ^bb2_block2v3(x2_v7: builtin.integer i64, y2_v8: builtin.integer i64) !9:
            y2_v11 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !10;
            z_v9 = math.i_add (x2_v7, y2_v11) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !11;
            branch.return z_v9 !12
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_does_not_fold_when_operands_are_nested_region_entry_args() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      test.test_region {
        ^region_entry(a: builtin.integer i64, b: builtin.integer i64):
        sum = math.i_add (a, b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return sum
      };
      done = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      branch.return done
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Unchanged);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            test.test_region 
            {
              ^region_entry_block2v1(a_v0: builtin.integer i64, b_v1: builtin.integer i64) !1:
                sum_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !2;
                branch.return sum_v2 !3
            } !4;
            done_v3 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !5;
            branch.return done_v3 !6
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_folds_inside_nested_region_using_outer_constant() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      outer_a = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      outer_b = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
      test.test_region {
        ^region_entry():
        inner_sum = math.i_add (outer_a, outer_b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return inner_sum
      };
      done = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      branch.return done
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            outer_a_v0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !1;
            outer_b_v1 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !2;
            test.test_region 
            {
              ^region_entry_block2v1() !3:
                inner_sum_v4 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !4;
                inner_sum_v2 = math.i_add (outer_a_v0, outer_b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !5;
                branch.return inner_sum_v4 !6
            } !7;
            done_v3 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !8;
            branch.return done_v3 !9
        }"#]].assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_folds_inside_two_nested_regions() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      test.test_two_regions {
        ^r0_entry():
        a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
        b0 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
        sum0 = math.i_add (a0, b0) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return sum0
      } {
        ^r1_entry():
        a1 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64;
        b1 = builtin.constant <builtin.integer <20: i64>> : builtin.integer i64;
        sum1 = math.i_add (a1, b1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return sum1
      };
      done = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      branch.return done
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    // Both inner adds should fold.
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            test.test_two_regions 
            {
              ^r0_entry_block2v1() !1:
                a0_v0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !2;
                b0_v1 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !3;
                sum0_v8 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !4;
                sum0_v2 = math.i_add (a0_v0, b0_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !5;
                branch.return sum0_v8 !6
            } 
            {
              ^r1_entry_block3v1() !7:
                a1_v3 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64 !8;
                b1_v4 = builtin.constant <builtin.integer <20: i64>> : builtin.integer i64 !9;
                sum1_v7 = builtin.constant <builtin.integer <30: i64>> : builtin.integer i64 !10;
                sum1_v5 = math.i_add (a1_v3, b1_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !11;
                branch.return sum1_v7 !12
            } !13;
            done_v6 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !14;
            branch.return done_v6 !15
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_folds_inside_nested_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      test.test_region {
        ^region_entry():
        a = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
        b = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
        inner_sum = math.i_add (a, b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return inner_sum
      };
      outer = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      branch.return outer
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    // The inner add should fold to 7.
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            test.test_region 
            {
              ^region_entry_block2v1() !1:
                a_v0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !2;
                b_v1 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !3;
                inner_sum_v4 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !4;
                inner_sum_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !5;
                branch.return inner_sum_v4 !6
            } !7;
            outer_v3 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !8;
            branch.return outer_v3 !9
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_materializes_constant_block_arg() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      c = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      cf.branch ^bb1(c)

      ^bb1(x: builtin.integer i64):
      branch.return x
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            c_v0 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !1;
            cf.branch ^bb1_block3v1(c_v0) !2

          ^bb1_block3v1(x_v1: builtin.integer i64) !3:
            x_v2 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !4;
            branch.return x_v2 !5
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_materializes_multiple_constant_block_args() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      b0 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
      a1 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      b1 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64;
      cf.branch_conditional if cond ^bb1(a0, b0) else ^bb1(a1, b1)

      ^bb1(x: builtin.integer i64, y: builtin.integer i64):
      branch.return x
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            a0_v1 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !1;
            b0_v2 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !2;
            a1_v3 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !3;
            b1_v4 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !4;
            cf.branch_conditional if cond_v0 ^bb1_block3v1(a0_v1, b0_v2) else ^bb1_block3v1(a1_v3, b1_v4) !5

          ^bb1_block3v1(x_v5: builtin.integer i64, y_v6: builtin.integer i64) !6:
            x_v7 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !7;
            y_v8 = builtin.constant <builtin.integer <5: i64>> : builtin.integer i64 !8;
            branch.return x_v7 !9
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_materializes_constant_carried_through_loop_back_edge() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      c = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      cf.branch ^loop(c)

      ^loop(x: builtin.integer i64):
      cf.branch_conditional if cond ^loop(x) else ^exit(x)

      ^exit(y: builtin.integer i64):
      branch.return y
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            c_v1 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !1;
            cf.branch ^loop_block3v1(c_v1) !2

          ^loop_block3v1(x_v2: builtin.integer i64) !3:
            x_v4 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !4;
            cf.branch_conditional if cond_v0 ^loop_block3v1(x_v4) else ^exit_block4v1(x_v4) !5

          ^exit_block4v1(y_v3: builtin.integer i64) !6:
            y_v5 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !7;
            branch.return y_v5 !8
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_loop_back_edge_with_different_constant_meets_to_not_a_constant() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      c1 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      cf.branch ^loop(c1)

      ^loop(x: builtin.integer i64):
      c2 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      cf.branch_conditional if cond ^loop(c2) else ^exit(x)

      ^exit(y: builtin.integer i64):
      branch.return y
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Unchanged);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            c1_v1 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !1;
            cf.branch ^loop_block3v1(c1_v1) !2

          ^loop_block3v1(x_v2: builtin.integer i64) !3:
            c2_v3 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !4;
            cf.branch_conditional if cond_v0 ^loop_block3v1(c2_v3) else ^exit_block4v1(x_v2) !5

          ^exit_block4v1(y_v4: builtin.integer i64) !6:
            branch.return y_v4 !7
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_materializes_constant_carried_through_loop_back_edge_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i32) -> (builtin.integer i64)> [] {
      ^entry(end: builtin.integer i32):
      start = builtin.constant <builtin.integer <0: i32>>: builtin.integer i32;
      step = builtin.constant <builtin.integer <1: i32>>: builtin.integer i32;
      c = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      y = scf.for start to end step step iter_args(c) {
        ^body(i: builtin.integer i32, c2: builtin.integer i64):
          branch.yield (c2)
      };
      branch.return y
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer i32) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(end_v0: builtin.integer i32) !0:
            start_v1 = builtin.constant <builtin.integer <0: i32>> : builtin.integer i32 !1;
            step_v2 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32 !2;
            c_v3 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !3;
            y_v4 = scf.for start_v1 to end_v0 step step_v2 iter_args(c_v3)
            {
              ^body_block2v1(i_v5: builtin.integer i32, c2_v6: builtin.integer i64) !4:
                c2_v8 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !5;
                branch.yield (c2_v8) !6
            } !7;
            y_v7 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !8;
            branch.return y_v7 !9
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_loop_back_edge_with_different_constant_meets_to_not_a_constant_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      c1 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      y = scf.while c1 : builtin.integer i64 {
        ^before(x: builtin.integer i64):
          branch.condition (cond, x)
      } do {
        ^after(x2: builtin.integer i64):
          c2 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
          branch.yield (c2)
      };
      branch.return y
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Unchanged);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            c1_v1 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !1;
            y_v5 = scf.while c1_v1 : builtin.integer i64
            {
              ^before_block2v1(x_v2: builtin.integer i64) !2:
                branch.condition (cond_v0, x_v2) !3
            } do 
            {
              ^after_block3v1(x2_v3: builtin.integer i64) !4:
                c2_v4 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !5;
                branch.yield (c2_v4) !6
            } !7;
            branch.return y_v5 !8
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_materialization_replaces_uses_of_block_arg() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      c = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
      cf.branch ^bb1(c)

      ^bb1(califragilistic: builtin.integer i64):
      sum = math.i_add (califragilistic, califragilistic) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
      branch.return sum
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            c_v0 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !1;
            cf.branch ^bb1_block3v1(c_v0) !2

          ^bb1_block3v1(califragilistic_v1: builtin.integer i64) !3:
            califragilistic_v4 = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64 !4;
            sum_v3 = builtin.constant <builtin.integer <84: i64>> : builtin.integer i64 !5;
            sum_v2 = math.i_add (califragilistic_v4, califragilistic_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !6;
            branch.return sum_v3 !7
        }"#]].assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_does_not_materialize_not_a_constant_block_arg() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool) -> (builtin.integer i64)> [] {
      ^entry(cond: cube.bool):
      a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      a1 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
      cf.branch_conditional if cond ^bb1(a0) else ^bb1(a1)

      ^bb1(x: builtin.integer i64):
      branch.return x
    }
  "#;

    let (status, after) = run_sccp_on_text(input)?;
    assert_eq!(status, IRStatus::Unchanged);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool ) -> (builtin.integer i64)> 
        {
          ^entry_block1v1(cond_v0: cube.bool ) !0:
            a0_v1 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !1;
            a1_v2 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64 !2;
            cf.branch_conditional if cond_v0 ^bb1_block3v1(a0_v1) else ^bb1_block3v1(a1_v2) !3

          ^bb1_block3v1(x_v3: builtin.integer i64) !4:
            branch.return x_v3 !5
        }"#]]
    .assert_eq(&after);
    Ok(())
}

#[test]
fn sccp_treats_free_variables_as_non_constant() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <() -> (builtin.integer i64)> [] {
      ^entry():
      outer_three = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
      outer_four = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
      test.test_region {
        ^region_entry():
        inner_sum = math.i_add (outer_three, outer_four) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return inner_sum
      };
      done = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64;
      branch.return done
    }
  "#;

    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let func_op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(func_op, ctx)?;

    use pliron::linked_list::ContainsLinkedList;
    let entry_block = func_op
        .deref(ctx)
        .regions()
        .next()
        .unwrap()
        .deref(ctx)
        .get_head()
        .unwrap();
    let region_op = entry_block
        .deref(ctx)
        .iter(ctx)
        .find(|op| Operation::get_opid(*op, ctx).to_string() == "test.test_region")
        .expect("test.test_region op should be in the entry block");

    let status = sccp(region_op, ctx)?;
    verify_operation(func_op, ctx)?;
    let after = Operation::get_op_dyn(func_op, ctx).disp(ctx).to_string();

    // Even though `outer_three` and `outer_four` are syntactically
    // `builtin.constant` ops, the analysis must treat them as NotAConstant when
    // they appear free inside the analysis root, so the inner `math.i_add` must
    // *not* fold.
    assert_eq!(status, IRStatus::Unchanged);
    expect![[r#"
        builtin.func @f: builtin.function <() -> (builtin.integer i64)> 
        {
          ^entry_block1v1() !0:
            outer_three_v0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64 !1;
            outer_four_v1 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !2;
            test.test_region 
            {
              ^region_entry_block2v1() !3:
                inner_sum_v2 = math.i_add (outer_three_v0, outer_four_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> !4;
                branch.return inner_sum_v2 !5
            } !6;
            done_v3 = builtin.constant <builtin.integer <99: i64>> : builtin.integer i64 !7;
            branch.return done_v3 !8
        }"#]].assert_eq(&after);
    Ok(())
}

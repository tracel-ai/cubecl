use cubecl_ir::{
    CanMaterialize, HasSideEffects, NoSideEffects,
    dialect::memory::AddressSpaceAttr,
    interfaces::{
        Deterministic,
        side_effects::{MemoryEffect, MemoryEffectsOp},
    },
    prelude::*,
};
use cubecl_opt::{
    analyses::dataflow_solver::pre::{AnticipatedMode, InsertMode},
    passes::pre::run_pre,
};
use expect_test::expect;
use pliron::{
    basic_block::BasicBlock, common_traits::Named,
    graph::walkers::uninterruptible::immutable::walk_op, init_env_logger_for_tests,
    irbuild::listener::DummyListener, irfmt::parsers::spaced, operation::verify_operation,
    parsable::parse_from_str, result::ExpectOk,
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

fn assert_pre_result(
    input: &str,
    antic: AnticipatedMode,
    insert: InsertMode,
    expected_status: IRStatus,
    check: impl FnOnce(&Context, Ptr<Operation>),
) -> Result<()> {
    init_env_logger_for_tests!();

    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(op, ctx)?;

    let before = op.dyn_op(ctx).disp(ctx).to_string();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let status = run_pre(ctx, op, &mut rewriter, antic, insert)?;

    verify_operation(op, ctx)?;

    let after = op.dyn_op(ctx).disp(ctx).to_string();
    assert_eq!(status, expected_status, "After PRE:\n{after}");

    if expected_status == IRStatus::Unchanged {
        assert_eq!(after, before);
    }

    check(ctx, op);
    Ok(())
}

fn assert_lcm_result(
    input: &str,
    expected_status: IRStatus,
    check: impl FnOnce(&Context, Ptr<Operation>),
) -> Result<()> {
    assert_pre_result(
        input,
        AnticipatedMode::Classical,
        InsertMode::Lazy,
        expected_status,
        check,
    )
}

fn assert_sscm_result(
    input: &str,
    expected_status: IRStatus,
    check: impl FnOnce(&Context, Ptr<Operation>),
) -> Result<()> {
    assert_pre_result(
        input,
        AnticipatedMode::Speculative,
        InsertMode::Sparse,
        expected_status,
        check,
    )
}

fn find_ops(ctx: &Context, root: Ptr<Operation>, name: &str) -> Vec<Ptr<Operation>> {
    let mut found = Vec::new();
    walk_op(
        ctx,
        &mut (&mut found, name),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (found, name), node| {
            if let IRNode::Operation(op) = node
                && &Operation::get_opid(op, ctx).to_string() == name
            {
                found.push(op);
            }
        },
    );
    found
}

fn find_block(ctx: &Context, root: Ptr<Operation>, name: &str) -> Ptr<BasicBlock> {
    let mut found = None;
    walk_op(
        ctx,
        &mut (&mut found, name),
        &WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, (found, name), node| {
            if let IRNode::BasicBlock(block) = node
                && block.given_name(ctx).is_some_and(|it| &it.as_ref() == name)
            {
                assert!(found.is_none(), "Duplicate block name: {name}");
                **found = Some(block);
            }
        },
    );
    found.unwrap_or_else(|| panic!("Missing block: {name}"))
}

fn find_ops_in(
    ctx: &Context,
    root: Ptr<Operation>,
    block_name: &str,
    op_name: &str,
) -> Vec<Ptr<Operation>> {
    let block = find_block(ctx, root, block_name);
    find_ops(ctx, root, op_name)
        .into_iter()
        .filter(|op| op.deref(ctx).get_parent_block() == Some(block))
        .collect()
}

#[track_caller]
fn find_single_op(ctx: &Context, root: Ptr<Operation>, name: &str) -> Ptr<Operation> {
    let ops = find_ops(ctx, root, name);
    assert_eq!(ops.len(), 1, "Expected one {name}");
    ops[0]
}

#[track_caller]
fn find_single_op_in(
    ctx: &Context,
    root: Ptr<Operation>,
    block_name: &str,
    op_name: &str,
) -> Ptr<Operation> {
    let ops = find_ops_in(ctx, root, block_name, op_name);
    assert_eq!(ops.len(), 1, "Expected one {op_name} in {block_name}");
    ops[0]
}

// Keep every expression observable so that DCE alone cannot satisfy the test.
#[test]
fn pre_no_redundancy_is_unchanged() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        builtin.integer ui32, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
        ab = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        ac = math.i_add (a, c) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box ab;
        pre.black_box ac;
        branch.return ac
    }
    "#;

    assert_lcm_result(input, IRStatus::Unchanged, |ctx, op| {
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 2);
        assert_eq!(find_ops(ctx, op, "pre.black_box").len(), 2);
    })
}

#[test]
fn pre_eliminates_commutative_and_dependent_duplicates() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        builtin.integer ui32, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
        ab = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        abc = math.i_add (ab, c) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        ba = math.i_add (b, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        bac = math.i_add (ba, c) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box ab;
        pre.black_box abc;
        pre.black_box ba;
        pre.black_box bac;
        branch.return bac
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(a_v0: builtin.integer ui32, b_v1: builtin.integer ui32, c_v2: builtin.integer ui32) !0:
            ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !1;
            abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !2;
            pre.black_box ab_v3 !3;
            pre.black_box abc_v4 !4;
            pre.black_box ab_v3 !5;
            pre.black_box abc_v4 !6;
            branch.return abc_v4 !7
        }"#]].assert_eq(&text);
    Ok(())
}

#[test]
fn pre_preserves_non_commutative_operand_order() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32):
        ab = math.u_div (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        ba = math.u_div (b, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        again = math.u_div (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box ab;
        pre.black_box ba;
        pre.black_box again;
        branch.return again
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(a_v0: builtin.integer ui32, b_v1: builtin.integer ui32) !0:
            ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !1;
            ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !2;
            pre.black_box ab_v2 !3;
            pre.black_box ba_v3 !4;
            pre.black_box ab_v2 !5;
            branch.return ab_v2 !6
        }"#]].assert_eq(&text);
    Ok(())
}

#[test]
fn pre_cfg_full_redundancy_adds_merge_argument() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.bool, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
        cf.branch_conditional if cond ^left() else ^right()

      ^left():
        left_sum = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box left_sum;
        cf.branch ^merge()

      ^right():
        right_sum = math.i_add (b, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box right_sum;
        cf.branch ^merge()

      ^merge():
        again = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.return again
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(cond_v0: cube.bool , a_v1: builtin.integer ui32, b_v2: builtin.integer ui32) !0:
            cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1() !1

          ^left_block4v1() !2:
            left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !3;
            pre.black_box left_sum_v3 !4;
            cf.branch ^merge_block3v3(left_sum_v3) !5

          ^right_block5v1() !6:
            right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !7;
            pre.black_box right_sum_v4 !8;
            cf.branch ^merge_block3v3(right_sum_v4) !9

          ^merge_block3v3(v6: builtin.integer ui32) !10:
            branch.return v6 !11
        }"#]].assert_eq(&text);
    Ok(())
}

#[test]
fn pre_cfg_partial_redundancy_inserts_dependent_expressions() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.bool, builtin.integer ui32, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
        cf.branch_conditional if cond ^left() else ^right()

      ^left():
        ab = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        abc = math.i_add (ab, c) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box abc;
        cf.branch ^merge(a)

      ^right():
        cf.branch ^merge(b)

      ^merge(saved: builtin.integer ui32):
        pre.black_box saved;
        ab_again = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        abc_again = math.i_add (ab_again, c) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.return abc_again
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(cond_v0: cube.bool , a_v1: builtin.integer ui32, b_v2: builtin.integer ui32, c_v3: builtin.integer ui32) !0:
            cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1() !1

          ^left_block4v1() !2:
            ab_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !3;
            abc_v5 = math.i_add (ab_v4, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !4;
            pre.black_box abc_v5 !5;
            cf.branch ^merge_block3v3(a_v1, abc_v5) !6

          ^right_block5v1() !7:
            ab_again_v9 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !8;
            abc_again_v10 = math.i_add (ab_again_v9, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !9;
            cf.branch ^merge_block3v3(b_v2, abc_again_v10) !10

          ^merge_block3v3(saved_v6: builtin.integer ui32, v11: builtin.integer ui32) !11:
            pre.black_box saved_v6 !12;
            branch.return v11 !13
        }"#]].assert_eq(&text);
    Ok(())
}

#[test]
fn pre_region_partial_redundancy_preserves_existing_result() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.bool, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
        selected = scf.if cond : builtin.integer ui32 then {
          ^then():
            sum = math.i_add (a, b) [] []:
                <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
            pre.black_box sum;
            branch.yield (a)
        } else {
          ^else():
            branch.yield (b)
        };
        pre.black_box selected;
        again = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.return again
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(cond_v0: cube.bool , a_v1: builtin.integer ui32, b_v2: builtin.integer ui32) !0:
            selected_v4, v7 = scf.if cond_v0 : builtin.integer ui32, builtin.integer ui32 then 
            {
              ^then_block2v1() !1:
                sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !2;
                pre.black_box sum_v3 !3;
                branch.yield (a_v1, sum_v3) !4
            } else 
            {
              ^else_block3v1() !5:
                again_v6 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !6;
                branch.yield (b_v2, again_v6) !7
            } !8;
            pre.black_box selected_v4 !9;
            branch.return v7 !10
        }"#]].assert_eq(&text);

    Ok(())
}

#[test]
fn pre_nested_regions_link_new_results() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.bool, cube.bool,
        builtin.integer ui32, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {
      ^entry(outer_cond: cube.bool, inner_cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
        outer = scf.if outer_cond : builtin.integer ui32 then {
          ^outer_then():
            inner = scf.if inner_cond : builtin.integer ui32 then {
              ^inner_then():
                sum = math.i_add (a, b) [] []:
                    <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
                pre.black_box sum;
                branch.yield (a)
            } else {
              ^inner_else():
                branch.yield (b)
            };
            pre.black_box inner;
            branch.yield (inner)
        } else {
          ^outer_else():
            branch.yield (c)
        };
        pre.black_box outer;
        again = math.i_add (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.return again
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , cube.bool , builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
        {
          ^entry_block1v1(outer_cond_v0: cube.bool , inner_cond_v1: cube.bool , a_v2: builtin.integer ui32, b_v3: builtin.integer ui32, c_v4: builtin.integer ui32) !0:
            outer_v7, v11 = scf.if outer_cond_v0 : builtin.integer ui32, builtin.integer ui32 then 
            {
              ^outer_then_block2v1() !1:
                inner_v6, v12 = scf.if inner_cond_v1 : builtin.integer ui32, builtin.integer ui32 then 
                {
                  ^inner_then_block3v1() !2:
                    sum_v5 = math.i_add (a_v2, b_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !3;
                    pre.black_box sum_v5 !4;
                    branch.yield (a_v2, sum_v5) !5
                } else 
                {
                  ^inner_else_block4v1() !6:
                    again_v9 = math.i_add (a_v2, b_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !7;
                    branch.yield (b_v3, again_v9) !8
                } !9;
                pre.black_box inner_v6 !10;
                branch.yield (inner_v6, v12) !11
            } else 
            {
              ^outer_else_block5v1() !12:
                again_v10 = math.i_add (a_v2, b_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !13;
                branch.yield (c_v4, again_v10) !14
            } !15;
            pre.black_box outer_v7 !16;
            branch.return v11 !17
        }"#]].assert_eq(&text);
    Ok(())
}

#[test]
fn pre_reuses_dominating_expression_across_loop() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.index, cube.index, cube.index,
        builtin.integer i64, builtin.integer i64
    ) -> ()>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
      ^entry(start: cube.index, end: cube.index, step: cube.index, a: builtin.integer i64, b: builtin.integer i64):
        before = math.i_add (a, b) [] []:
            <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        pre.black_box before;
        scf.for start to end step step iter_args() {
          ^body(iv: cube.index):
            inside = math.i_add (a, b) [] []:
                <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
            pre.black_box inside;
            branch.yield ()
        };
        after = math.i_add (a, b) [] []:
            <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        pre.black_box after;
        branch.return
    }
    "#;

    assert_lcm_result(input, IRStatus::Changed, |ctx, op| {
        let sum = find_single_op_in(ctx, op, "entry", "math.i_add").result(ctx);
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert_eq!(find_ops(ctx, op, "scf.for").len(), 1);

        let uses = find_ops(ctx, op, "pre.black_box");
        assert_eq!(uses.len(), 3);
        for use_op in uses {
            assert_eq!(use_op.operand(ctx, 0), sum);
        }
    })
}

#[test]
fn pre_loop_carried_expression_stays_in_loop() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(
        cube.index, cube.index, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
      ^entry(start: cube.index, step: cube.index, init: builtin.integer ui32, b: builtin.integer ui32):
        end = cube.read_builtin (UnitPos) : cube.index;
        result = scf.for start to end step step iter_args(init) {
          ^body(iv: cube.index, acc: builtin.integer ui32):
            first = math.i_add (acc, b) [] []:
                <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
            second = math.i_add (acc, b) [] []:
                <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
            pre.black_box first;
            pre.black_box second;
            branch.yield (second)
        };
        branch.return result
    }
    "#;

    let (_, text) = run_lcm_on_text(input)?;
    expect![[r#"
        builtin.func @f: builtin.function <(cube.index , cube.index , builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]
        {
          ^entry_block1v1(start_v0: cube.index , step_v1: cube.index , init_v2: builtin.integer ui32, b_v3: builtin.integer ui32) !0:
            end_v4 = cube.read_builtin (UnitPos) : cube.index  !1;
            result_v5 = scf.for start_v0 to end_v4 step step_v1 iter_args(init_v2) 
            {
              ^body_block2v1(iv_v6: cube.index , acc_v7: builtin.integer ui32) !2:
                first_v8 = math.i_add (acc_v7, b_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> !3;
                pre.black_box first_v8 !4;
                pre.black_box first_v8 !5;
                branch.yield (first_v8) !6
            } !7;
            branch.return result_v5 !8
        }"#]].assert_eq(&text);

    assert_sscm_result(input, IRStatus::Changed, |ctx, op| {
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert!(find_ops_in(ctx, op, "entry", "math.i_add").is_empty());

        let body = find_block(ctx, op, "body");
        assert_eq!(body.deref(ctx).get_num_arguments(), 2);

        let sum = find_single_op_in(ctx, op, "body", "math.i_add");
        assert_eq!(sum.operand(ctx, 0), body.deref(ctx).get_argument(1));

        let uses = find_ops_in(ctx, op, "body", "pre.black_box");
        assert_eq!(uses.len(), 2);
        for use_op in uses {
            assert_eq!(use_op.operand(ctx, 0), sum.result(ctx));
        }
        assert_eq!(
            find_single_op_in(ctx, op, "body", "branch.yield").operands(ctx),
            vec![sum.result(ctx)],
        );

        let loop_op = find_single_op(ctx, op, "scf.for");
        assert_eq!(loop_op.results(ctx).len(), 1);
        assert_eq!(
            find_single_op(ctx, op, "branch.return").operand(ctx, 0),
            loop_op.result(ctx),
        );
    })
}

// Function arguments are uniform because this is an entry point.
// plane.elect supplies an explicitly divergent condition.
fn pre_branch_motion_input(divergent: bool, then_body: &str) -> String {
    let condition_setup = if divergent {
        "cond = plane.elect () [] []: <() -> (cube.bool)>;"
    } else {
        ""
    };
    let condition = if divergent { "cond" } else { "uniform_cond" };

    format!(
        r#"
  builtin.module @module {{
    ^module_block():
    builtin.func @f: builtin.function <(
        cube.bool, builtin.integer ui32, builtin.integer ui32
    ) -> (builtin.integer ui32)>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {{
      ^entry(uniform_cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
        {condition_setup}
        result = scf.if {condition} : builtin.integer ui32 then {{
          ^then():
            {then_body}
            branch.yield (value)
        }} else {{
          ^else():
            branch.yield (a)
        }};
        branch.return result
    }}
  }}
    "#
    )
}

fn pre_branch_add_body() -> &'static str {
    r#"
    value = math.i_add (a, b) [] []:
        <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
    "#
}

#[test]
fn pre_uniform_branch_does_not_speculate() -> Result<()> {
    let input = pre_branch_motion_input(false, pre_branch_add_body());

    assert_sscm_result(&input, IRStatus::Unchanged, |ctx, op| {
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert!(find_ops_in(ctx, op, "entry", "math.i_add").is_empty());

        let sum = find_single_op_in(ctx, op, "then", "math.i_add").result(ctx);
        assert_eq!(
            find_single_op_in(ctx, op, "then", "branch.yield").operands(ctx),
            vec![sum],
        );
    })
}

#[test]
fn pre_divergent_branch_requires_speculative_mode() -> Result<()> {
    let input = pre_branch_motion_input(true, pre_branch_add_body());

    // Classical lazy motion must retain the computation in the taken branch.
    assert_lcm_result(&input, IRStatus::Unchanged, |ctx, op| {
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert_eq!(find_ops_in(ctx, op, "then", "math.i_add").len(), 1);
        assert!(find_ops_in(ctx, op, "entry", "math.i_add").is_empty());
    })?;

    // Speculative sparse motion may compute it before divergent control flow.
    assert_sscm_result(&input, IRStatus::Changed, |ctx, op| {
        let sum = find_single_op_in(ctx, op, "entry", "math.i_add").result(ctx);
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert!(find_ops_in(ctx, op, "then", "math.i_add").is_empty());
        assert_eq!(
            find_single_op_in(ctx, op, "then", "branch.yield").operands(ctx),
            vec![sum],
        );

        let entry = find_block(ctx, op, "entry");
        assert_eq!(
            find_single_op_in(ctx, op, "else", "branch.yield").operands(ctx),
            vec![entry.deref(ctx).get_argument(1)],
        );
        assert_eq!(
            find_single_op(ctx, op, "branch.return").operand(ctx, 0),
            find_single_op(ctx, op, "scf.if").result(ctx),
        );
    })
}

#[test]
fn pre_does_not_speculate_division_or_its_dependents() -> Result<()> {
    let input = pre_branch_motion_input(
        true,
        r#"
        quotient = math.u_div (a, b) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        value = math.i_add (quotient, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        "#,
    );

    // b is unknown and may be zero. Neither the division nor an expression
    // depending on its result may be speculated onto the other branch.
    assert_sscm_result(&input, IRStatus::Unchanged, |ctx, op| {
        let division = find_single_op_in(ctx, op, "then", "math.u_div");
        let sum = find_single_op_in(ctx, op, "then", "math.i_add");

        assert_eq!(find_ops(ctx, op, "math.u_div").len(), 1);
        assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
        assert!(find_ops_in(ctx, op, "entry", "math.u_div").is_empty());
        assert!(find_ops_in(ctx, op, "entry", "math.i_add").is_empty());
        assert_eq!(sum.operand(ctx, 0), division.result(ctx));
        assert_eq!(
            find_single_op_in(ctx, op, "then", "branch.yield").operands(ctx),
            vec![sum.result(ctx)],
        );
    })
}

fn pre_loop_motion_input(divergent: bool) -> String {
    let bound_setup = if divergent {
        "end = cube.read_builtin (UnitPos) : cube.index;"
    } else {
        ""
    };
    let end = if divergent { "end" } else { "uniform_end" };

    format!(
        r#"
  builtin.module @module {{
    ^module_block():
    builtin.func @f: builtin.function <(
        cube.index, cube.index, cube.index,
        builtin.integer i64, builtin.integer i64
    ) -> ()>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {{
      ^entry(start: cube.index, uniform_end: cube.index, step: cube.index, a: builtin.integer i64, b: builtin.integer i64):
        {bound_setup}
        scf.for start to {end} step step iter_args() {{
          ^body(iv: cube.index):
            value = math.i_add (a, b) [] []:
                <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
            pre.black_box value;
            branch.yield ()
        }};
        branch.return
    }}
  }}
    "#
    )
}

#[test]
fn pre_loop_speculation_depends_on_bound_uniformity() -> Result<()> {
    for divergent in [false, true] {
        let input = pre_loop_motion_input(divergent);
        let expected = if divergent {
            IRStatus::Changed
        } else {
            IRStatus::Unchanged
        };

        assert_sscm_result(&input, expected, |ctx, op| {
            assert_eq!(find_ops(ctx, op, "math.i_add").len(), 1);
            assert_eq!(find_ops(ctx, op, "scf.for").len(), 1);

            let expected_block = if divergent { "entry" } else { "body" };
            let sum = find_single_op_in(ctx, op, expected_block, "math.i_add").result(ctx);
            assert_eq!(
                find_single_op_in(ctx, op, "body", "pre.black_box").operand(ctx, 0),
                sum,
            );

            let other_block = if divergent { "body" } else { "entry" };
            assert!(find_ops_in(ctx, op, other_block, "math.i_add").is_empty());
        })?;
    }
    Ok(())
}

// Memory fixtures follow the address-space fixtures in pre_analyses.rs.
// The read explicitly implements Deterministic so value numbering can merge
// reads of the same memory version. Writes remain observable side effects.

#[pliron_op(
    name = "pre.test_read",
    format = "attr($read_address_space, $AddressSpaceAttr) `:` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (read_address_space: AddressSpaceAttr)
)]
#[op_interfaces(Deterministic)]
#[op_traits(CanMaterialize, NoSideEffects)]
pub struct PreTestRead;

#[op_interface_impl]
impl MemoryEffectsOp for PreTestRead {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_read_address_space(ctx).unwrap().0;
        vec![MemoryEffect::ReadAllInSpace(space)]
    }
}

#[pliron_op(
    name = "pre.test_write",
    format = "attr($write_address_space, $AddressSpaceAttr)",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (write_address_space: AddressSpaceAttr)
)]
#[op_traits(HasSideEffects)]
pub struct PreTestWrite;

#[op_interface_impl]
impl MemoryEffectsOp for PreTestWrite {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_write_address_space(ctx).unwrap().0;
        vec![MemoryEffect::WriteAllInSpace(space)]
    }
}

fn pre_memory_input(write_space: &str) -> String {
    format!(
        r#"
    builtin.func @f: builtin.function <(
        builtin.integer ui32
    ) -> (builtin.integer ui32)> [] {{
      ^entry(a: builtin.integer ui32):
        before = pre.test_read Shared : builtin.integer ui32;
        before_sum = math.i_add (before, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.test_write {write_space};
        after = pre.test_read Shared : builtin.integer ui32;
        after_sum = math.i_add (after, a) [] []:
            <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        pre.black_box before_sum;
        pre.black_box after_sum;
        branch.return after_sum
    }}
    "#
    )
}

#[test]
fn pre_reuses_reads_and_dependents_across_non_aliasing_write() -> Result<()> {
    let input = pre_memory_input("Local");

    assert_lcm_result(&input, IRStatus::Changed, |ctx, op| {
        let read = find_single_op(ctx, op, "pre.test_read");
        let sum = find_single_op(ctx, op, "math.i_add");
        assert_eq!(find_ops(ctx, op, "pre.test_write").len(), 1);
        assert_eq!(sum.operand(ctx, 0), read.result(ctx));

        let uses = find_ops(ctx, op, "pre.black_box");
        assert_eq!(uses.len(), 2);
        for use_op in uses {
            assert_eq!(use_op.operand(ctx, 0), sum.result(ctx));
        }
        assert_eq!(
            find_single_op(ctx, op, "branch.return").operand(ctx, 0),
            sum.result(ctx),
        );
    })
}

#[test]
fn pre_aliasing_write_preserves_reads_and_dependents() -> Result<()> {
    let input = pre_memory_input("Shared");

    assert_lcm_result(&input, IRStatus::Unchanged, |ctx, op| {
        let reads = find_ops(ctx, op, "pre.test_read");
        let sums = find_ops(ctx, op, "math.i_add");
        let uses = find_ops(ctx, op, "pre.black_box");

        assert_eq!(reads.len(), 2);
        assert_eq!(sums.len(), 2);
        assert_eq!(uses.len(), 2);
        assert_eq!(find_ops(ctx, op, "pre.test_write").len(), 1);

        for i in 0..2 {
            assert_eq!(sums[i].operand(ctx, 0), reads[i].result(ctx));
            assert_eq!(uses[i].operand(ctx, 0), sums[i].result(ctx));
        }
        assert_ne!(uses[0].operand(ctx, 0), uses[1].operand(ctx, 0));
        assert_eq!(
            find_single_op(ctx, op, "branch.return").operand(ctx, 0),
            sums[1].result(ctx),
        );
    })
}

#[test]
fn pre_is_idempotent_after_speculative_insertion() -> Result<()> {
    init_env_logger_for_tests!();

    let ctx = &mut Context::new();
    let op = parse_from_str(
        spaced(Operation::top_level_parser()),
        ctx,
        sscm_paper_example(),
    )
    .expect_ok(ctx);
    verify_operation(op, ctx)?;

    let mut rewriter = IRRewriter::<DummyListener>::default();

    let first = run_pre(
        ctx,
        op,
        &mut rewriter,
        AnticipatedMode::Speculative,
        InsertMode::Sparse,
    )?;
    assert_eq!(first, IRStatus::Changed);
    verify_operation(op, ctx)?;
    let once = op.dyn_op(ctx).disp(ctx).to_string();

    let second = run_pre(
        ctx,
        op,
        &mut rewriter,
        AnticipatedMode::Speculative,
        InsertMode::Sparse,
    )?;
    assert_eq!(second, IRStatus::Unchanged);
    verify_operation(op, ctx)?;

    let twice = op.dyn_op(ctx).disp(ctx).to_string();
    assert_eq!(twice, once);

    Ok(())
}

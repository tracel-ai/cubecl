use cubecl_ir::prelude::*;
use cubecl_opt::passes::mem2reg::Mem2RegPass;
use expect_test::expect;
use pliron::{
    init_env_logger_for_tests,
    irfmt::parsers::spaced,
    operation::verify_operation,
    parsable::parse_from_str,
    pass::{AnalysisManager, Pass},
    result::ExpectOk,
};

fn run_mem2reg_on_text(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;
    Mem2RegPass.run(op, ctx, &mut AnalysisManager::default())?;
    verify_operation(op, ctx)?;

    Ok(Operation::get_op_dyn(op, ctx).disp(ctx).to_string())
}

/// `@f` taking `params` and returning `results`, with `LOCAL` in `body` declaring a local
/// `cube.index` and `LOAD` and `STORE` typing its loads and stores. The parser of
/// `memory.declare_variable` requires an `init`.
fn func(params: &str, results: &str, body: &str) -> String {
    let param_types: Vec<_> = params
        .split(", ")
        .map(|param| param.split_once(": ").unwrap().1)
        .collect();
    let body = body
        .replace(
            "LOCAL",
            "memory.declare_variable cube.index Local, align = 4, init = cube.index 0",
        )
        .replace(
            "LOAD",
            "[] []: <(cube.ptr <cube.index, Local>) -> (cube.index)>",
        )
        .replace(
            "STORE",
            "[] []: <(cube.ptr <cube.index, Local>, cube.index) -> ()>",
        );
    format!(
        "builtin.func @f: builtin.function <({}) -> ({results})> [] {{\n  ^entry({params}):\n{body}\n}}",
        param_types.join(", ")
    )
}

#[test]
fn mem2reg_promotes_interleaved_allocs_of_one_block() -> Result<()> {
    let input = func(
        "x: cube.index, y: cube.index",
        "cube.index, cube.index",
        r#"
        a = LOCAL;
        b = LOCAL;
        memory.store (a, x) STORE;
        b0 = memory.load (b) LOAD;
        memory.store (b, y) STORE;
        memory.store (a, b0) STORE;
        a1 = memory.load (a) LOAD;
        b1 = memory.load (b) LOAD;
        branch.return a1, b1
        "#,
    );

    let ctx = &mut Context::default();
    let printed = run_mem2reg_on_text(ctx, &input)?;

    expect![[r#"
        builtin.func @f: builtin.function <(cube.index , cube.index ) -> (cube.index , cube.index )> 
        {
          ^entry_block1v1(x_v0: cube.index , y_v1: cube.index ) !0:
            b0_v8 = builtin.constant <cube.index 0> : cube.index  !1;
            v7 = builtin.constant <cube.index 0> : cube.index ;
            branch.return b0_v8, y_v1 !2
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn mem2reg_threads_a_store_out_of_one_branch_of_an_if() -> Result<()> {
    let input = func(
        "cond: cube.bool, x: cube.index, y: cube.index",
        "cube.index, cube.index",
        r#"
        a = LOCAL;
        memory.store (a, x) STORE;
        z = scf.if cond : cube.index then {
          ^then():
            memory.store (a, y) STORE;
            branch.yield (y)
        } else {
          ^else():
            branch.yield (x)
        };
        l = memory.load (a) LOAD;
        branch.return l, z
        "#,
    );

    let ctx = &mut Context::default();
    let printed = run_mem2reg_on_text(ctx, &input)?;

    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , cube.index , cube.index ) -> (cube.index , cube.index )> 
        {
          ^entry_block1v1(cond_v0: cube.bool , x_v1: cube.index , y_v2: cube.index ) !0:
            v7 = builtin.constant <cube.index 0> : cube.index ;
            z_v4, l_v6 = scf.if cond_v0 : cube.index , cube.index  then 
            {
              ^then_block2v1() !1:
                branch.yield (y_v2, y_v2) !2
            } else 
            {
              ^else_block3v1() !3:
                branch.yield (x_v1, x_v1) !4
            } !5;
            branch.return l_v6, z_v4 !6
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn mem2reg_threads_stores_out_of_both_branches_of_an_if() -> Result<()> {
    let input = func(
        "cond: cube.bool, x: cube.index, y: cube.index",
        "cube.index, cube.index",
        r#"
        a = LOCAL;
        z = scf.if cond : cube.index then {
          ^then():
            memory.store (a, x) STORE;
            branch.yield (y)
        } else {
          ^else():
            memory.store (a, y) STORE;
            branch.yield (x)
        };
        l = memory.load (a) LOAD;
        branch.return l, z
        "#,
    );

    let ctx = &mut Context::default();
    let printed = run_mem2reg_on_text(ctx, &input)?;

    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , cube.index , cube.index ) -> (cube.index , cube.index )> 
        {
          ^entry_block1v1(cond_v0: cube.bool , x_v1: cube.index , y_v2: cube.index ) !0:
            v6 = builtin.constant <cube.index 0> : cube.index ;
            z_v4, l_v7 = scf.if cond_v0 : cube.index , cube.index  then 
            {
              ^then_block2v1() !1:
                branch.yield (y_v2, x_v1) !2
            } else 
            {
              ^else_block3v1() !3:
                branch.yield (x_v1, y_v2) !4
            } !5;
            branch.return l_v7, z_v4 !6
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn mem2reg_carries_a_store_through_a_for_loop() -> Result<()> {
    let input = func(
        "x: cube.index, end: cube.index",
        "cube.index",
        r#"
        start = builtin.constant <cube.index 0> : cube.index;
        step = builtin.constant <cube.index 1> : cube.index;
        a = LOCAL;
        memory.store (a, x) STORE;
        scf.for start to end step step iter_args() {
          ^body(i: cube.index):
            l = memory.load (a) LOAD;
            memory.store (a, i) STORE;
            branch.yield ()
        };
        r = memory.load (a) LOAD;
        branch.return r
        "#,
    );

    let ctx = &mut Context::default();
    let printed = run_mem2reg_on_text(ctx, &input)?;

    expect![[r#"
        builtin.func @f: builtin.function <(cube.index , cube.index ) -> (cube.index )> 
        {
          ^entry_block1v1(x_v0: cube.index , end_v1: cube.index ) !0:
            v10 = builtin.constant <cube.index 0> : cube.index ;
            start_v2 = builtin.constant <cube.index 0> : cube.index  !1;
            step_v3 = builtin.constant <cube.index 1> : cube.index  !2;
            r_v9 = scf.for start_v2 to end_v1 step step_v3 iter_args(x_v0) 
            {
              ^body_block2v1(i_v5: cube.index , l_v8: cube.index ) !3:
                branch.yield (i_v5) !4
            } !5;
            branch.return r_v9 !6
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

#[test]
fn mem2reg_merges_stores_of_two_cfg_paths() -> Result<()> {
    let input = func(
        "cond: cube.bool, x: cube.index, y: cube.index",
        "cube.index",
        r#"
        a = LOCAL;
        memory.store (a, x) STORE;
        cf.branch_conditional if cond ^bb0() else ^bb1()

      ^bb0():
        memory.store (a, y) STORE;
        cf.branch ^bb2()

      ^bb1():
        cf.branch ^bb2()

      ^bb2():
        l = memory.load (a) LOAD;
        branch.return l
        "#,
    );

    let ctx = &mut Context::default();
    let printed = run_mem2reg_on_text(ctx, &input)?;

    expect![[r#"
        builtin.func @f: builtin.function <(cube.bool , cube.index , cube.index ) -> (cube.index )> 
        {
          ^entry_block1v1(cond_v0: cube.bool , x_v1: cube.index , y_v2: cube.index ) !0:
            v6 = builtin.constant <cube.index 0> : cube.index ;
            cf.branch_conditional if cond_v0 ^bb0_block4v1() else ^bb1_block5v1() !1

          ^bb0_block4v1() !2:
            cf.branch ^bb2_block3v3(y_v2) !3

          ^bb1_block5v1() !4:
            cf.branch ^bb2_block3v3(x_v1) !5

          ^bb2_block3v3(l_v5: cube.index ) !6:
            branch.return l_v5 !7
        }"#]]
    .assert_eq(&printed);
    Ok(())
}

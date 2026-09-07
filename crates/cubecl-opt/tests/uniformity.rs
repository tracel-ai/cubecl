use cubecl_ir::{
    ident,
    interfaces::uniformity::Uniformity,
    prelude::{IRNode, SingleBlockRegionInterface},
    rewrite::{WALKCONFIG_ANY, visit_all_values},
};
use cubecl_opt::{
    analyses::dataflow_solver::{
        DataflowSolver,
        control_flow_uniformity::BlockUniformity,
        value_uniformity::{DynamicUniformityLattice, StrictUniformityLattice},
    },
    passes::uniformity::uniformity_solver,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    common_traits::Named,
    context::{Context, Ptr},
    graph::walkers::uninterruptible::immutable::walk_op,
    init_env_logger_for_tests,
    irfmt::parsers::spaced,
    op::Op,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    result::{ExpectOk, Result},
    value::Value,
};

fn run_uniformity_on_text(ctx: &mut Context, input: &str) -> Result<(ModuleOp, DataflowSolver)> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;
    let module = ModuleOp::new(ctx, ident("module"));
    op.insert_at_front(module.get_body(ctx, 0), ctx);

    let solver = uniformity_solver(module.get_operation(), ctx)?;
    Ok((module, solver))
}

#[track_caller]
fn value(ctx: &Context, module: ModuleOp, name: &str) -> Value {
    let op = module.get_operation();
    let mut out_value = None;
    visit_all_values(
        ctx,
        &mut (&mut out_value, name),
        op,
        |ctx, (out_value, name), val| {
            if val.given_name(ctx).is_some_and(|it| it.as_ref() == *name) {
                **out_value = Some(val);
            }
        },
    );
    out_value.expect("Can't find value")
}

#[track_caller]
fn block(ctx: &Context, module: ModuleOp, name: &str) -> Ptr<BasicBlock> {
    let op = module.get_operation();
    let mut out_block = None;
    walk_op(
        ctx,
        &mut (&mut out_block, name),
        &WALKCONFIG_ANY,
        op,
        |ctx, (out_block, name), node| {
            if let IRNode::BasicBlock(block) = node
                && block.given_name(ctx).is_some_and(|it| it.as_ref() == *name)
            {
                **out_block = Some(block);
            }
        },
    );
    out_block.expect("Can't find block")
}

#[track_caller]
fn assert_dynamic_uniformity(solver: &DataflowSolver, value: Value, expected: Uniformity) {
    let actual = solver.lookup_state::<DynamicUniformityLattice>(value);
    let actual = actual.expect("not found").deref().value().0;
    assert_eq!(actual, expected, "Mismatched dynamic uniformity");
}

#[track_caller]
fn assert_strict_uniformity(solver: &DataflowSolver, value: Value, expected: Uniformity) {
    let actual = solver.lookup_state::<StrictUniformityLattice>(value);
    let actual = actual.expect("not found").deref().value().0;
    assert_eq!(actual, expected, "Mismatched strict uniformity");
}

#[track_caller]
fn assert_block_uniformity(solver: &DataflowSolver, block: Ptr<BasicBlock>, expected: Uniformity) {
    let actual = solver.lookup_state::<BlockUniformity>(block);
    let actual = actual.expect("not found").deref().value();
    assert_eq!(actual, expected, "Mismatched block uniformity");
}

fn kernel(content: &str) -> String {
    format!(
        r#"
    builtin.func @f: builtin.function <() -> ()>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {{
      ^entry():
        {content}
        branch.return
    }}
  "#
    )
}

#[test]
fn uniformity_propagating_op_propagates() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
    x = cube.buffer_len buffer_0 : cube.index;
    y = math.i_add (x, x) [] [] : <(cube.index, cube.index) -> (cube.index)>;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Device);
    assert_dynamic_uniformity(&solver, value(ctx, module, "y"), Uniformity::Device);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);
    assert_strict_uniformity(&solver, value(ctx, module, "y"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_constant_is_intrinsically_uniform() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = builtin.constant <builtin.integer <42: i64>> : builtin.integer i64;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Device);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_cube_pos_builtin_read_has_expected_uniformity() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.read_builtin (CubePos) : cube.index;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_plane_pos_builtin_read_has_expected_uniformity() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.read_builtin (PlanePos) : cube.index;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Plane);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Plane);

    Ok(())
}

#[test]
fn uniformity_unit_pos_builtin_read_has_expected_uniformity() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.read_builtin (UnitPos) : cube.index;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::None);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::None);

    Ok(())
}

#[test]
fn uniformity_metadata_is_always_device_uniform() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.buffer_len buffer_0 : cube.index;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Device);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_plane_elect_is_never_uniform() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = plane.elect () [] []: <() -> (cube.bool)>;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::None);

    Ok(())
}

#[test]
fn uniformity_plane_sum_is_at_least_plane_uniform() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.read_builtin (UnitPos) : cube.index;
      y = plane.i_sum (x) [] [] : <(cube.index) -> (cube.index)>;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "y"), Uniformity::Plane);

    Ok(())
}

#[test]
fn uniformity_plane_sum_inherits_if_larger_than_plane_uniform() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      x = cube.read_builtin (CubePos) : cube.index;
      y = plane.i_sum (x) [] [] : <(cube.index) -> (cube.index)>;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "y"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_if_uniform_condition_preserves_uniformity() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      cond = cube.read_scalar <cube.bool>[0] : cube.bool;
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
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Device);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_if_divergent_condition_makes_propagating_result_divergent() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      index = cube.read_builtin (PlanePos) : cube.index;
      zero = builtin.constant <cube.index 0> : cube.index;
      cond = cmp.i_equal (index, zero) [] [] : <(cube.index, cube.index) -> (cube.bool)>;
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
      z = math.i_add (x, y) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Plane);
    assert_strict_uniformity(&solver, value(ctx, module, "x"), Uniformity::Plane);

    assert_dynamic_uniformity(&solver, value(ctx, module, "z"), Uniformity::Plane);
    assert_strict_uniformity(&solver, value(ctx, module, "z"), Uniformity::Plane);

    Ok(())
}

#[test]
fn uniformity_if_dead_branch_inherits_live_branch_uniformity() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      cond = builtin.constant <cube.bool true> : cube.bool;
      x = scf.if cond : builtin.integer i64 then {
        ^then_block():
          a0 = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
          branch.yield (a0)
      } else {
        ^else_block():
          a1 = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
          branch.yield (a1)
      };
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_dynamic_uniformity(&solver, value(ctx, module, "x"), Uniformity::Device);

    Ok(())
}

#[test]
fn uniformity_if_divergence_does_not_escape_to_parent_block() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      index = cube.read_builtin (PlanePos) : cube.index;
      zero = builtin.constant <cube.index 0> : cube.index;
      cond = cmp.i_equal (index, zero) [] [] : <(cube.index, cube.index) -> (cube.bool)>;
      x = scf.if cond : cube.index then {
        ^then_block():
          a0 = cube.read_builtin (UnitPos) : cube.index;
          branch.yield (a0)
      } else {
        ^else_block():
          a1 = cube.read_builtin (AbsolutePos) : cube.index;
          branch.yield (a1)
      };
      z = builtin.constant <builtin.integer <7: i64>> : builtin.integer i64;
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_block_uniformity(&solver, block(ctx, module, "entry"), Uniformity::Cube);
    assert_block_uniformity(&solver, block(ctx, module, "then_block"), Uniformity::Plane);
    assert_block_uniformity(&solver, block(ctx, module, "else_block"), Uniformity::Plane);

    assert_dynamic_uniformity(&solver, value(ctx, module, "z"), Uniformity::Device);
    assert_strict_uniformity(&solver, value(ctx, module, "z"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_for_uniform_start_end_step() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      start = cube.read_builtin (CubePos) : cube.index;
      end = cube.read_builtin (CubeCount) : cube.index;
      step = builtin.constant <cube.index 1> : cube.index;
      c = cube.read_scalar <cube.bool>[0] : cube.bool;
      y = scf.for start to end step step iter_args(c) {
        ^body(i: cube.index, c2: cube.bool):
          x = cube.bool_and (c2, c2) [] [] : <(cube.bool, cube.bool) -> (cube.bool)>;
          branch.yield (x)
      };
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_block_uniformity(&solver, block(ctx, module, "body"), Uniformity::Cube);
    assert_dynamic_uniformity(&solver, value(ctx, module, "y"), Uniformity::Cube);

    Ok(())
}

#[test]
fn uniformity_for_divergent_start_makes_body_divergent() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      start = cube.read_builtin (PlanePos) : cube.index;
      end = cube.read_builtin (CubeDim) : cube.index;
      step = builtin.constant <cube.index 1> : cube.index;
      scf.for start to end step step iter_args() {
        ^body(i: builtin.integer i32):
          branch.yield ()
      };
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_block_uniformity(&solver, block(ctx, module, "body"), Uniformity::Plane);

    Ok(())
}

#[test]
fn uniformity_for_divergent_end_makes_body_divergent() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      start = cube.read_builtin (CubePos) : cube.index;
      end = cube.read_builtin (PlanePos) : cube.index;
      step = builtin.constant <cube.index 1> : cube.index;
      scf.for start to end step step iter_args() {
        ^body(i: builtin.integer i32):
          branch.yield ()
      };
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_block_uniformity(&solver, block(ctx, module, "body"), Uniformity::Plane);

    Ok(())
}

#[test]
fn uniformity_for_divergent_step_makes_body_divergent() -> Result<()> {
    let ctx = &mut Context::new();
    let input = kernel(
        r#"
      start = cube.read_builtin (CubePos) : cube.index;
      end = cube.read_builtin (CubeCount) : cube.index;
      step = cube.read_builtin (PlanePos) : cube.index;
      scf.for start to end step step iter_args() {
        ^body(i: builtin.integer i32):
          branch.yield ()
      };
    "#,
    );

    let (module, solver) = run_uniformity_on_text(ctx, &input)?;

    assert_block_uniformity(&solver, block(ctx, module, "body"), Uniformity::Plane);

    Ok(())
}

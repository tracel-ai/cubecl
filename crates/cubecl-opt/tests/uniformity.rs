use cubecl_ir::{ident, prelude::SingleBlockRegionInterface};
use cubecl_opt::passes::uniformity::uniformity;
use pliron::{
    builtin::ops::ModuleOp,
    context::Context,
    init_env_logger_for_tests,
    irbuild::IRStatus,
    irfmt::parsers::spaced,
    op::Op,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    result::{ExpectOk, Result},
};

fn run_uniformity_on_text(input: &str) -> Result<(IRStatus, String)> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;
    let module = ModuleOp::new(ctx, ident("module"));
    op.insert_at_front(module.get_body(ctx, 0), ctx);

    let status = uniformity(module.get_operation(), ctx)?;

    let after = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();
    log::trace!("After SCCP:\n{}", after);
    verify_operation(op, ctx)?;
    Ok((status, after))
}

#[test]
fn uniformity_is_path_sensitive_region() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer i64) -> (builtin.integer i64)>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
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

    let (status, after) = run_uniformity_on_text(input)?;
    assert_eq!(status, IRStatus::Changed);
    todo!()
}

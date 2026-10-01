//! Integer `x % 1` folds to zero; floating-point `x % 1.0` stays fractional.

use cubecl_ir::{ConstantValue, interfaces::ConstantAttr, prelude::*, rewrite::SimplifyOps};
use cubecl_opt::passes::sccp::sccp;
use pliron::{
    attribute::attr_cast,
    builtin::ops::{ConstantOp, FuncOp},
    init_env_logger_for_tests,
    irbuild::match_rewrite::{RewriterOrder, apply_match_rewrite},
    irfmt::parsers::spaced,
    operation::verify_operation,
    parsable::parse_from_str,
    result::ExpectOk,
};

#[test]
fn remainder_by_one_preserves_semantics() -> Result<()> {
    init_env_logger_for_tests!();
    for (rem, ty, one, expected) in [
        (
            "u_rem",
            "cube.index",
            "cube.index 1",
            Some(ConstantValue::UInt(0)),
        ),
        (
            "s_rem",
            "builtin.integer si64",
            "builtin.integer <1: si64>",
            Some(ConstantValue::Int(0)),
        ),
        ("f_rem", "cube.f32", "cube.float cube.f32: 1.0", None),
    ] {
        let input = format!(
            r#"
            builtin.func @f: builtin.function <({ty}) -> ({ty})> [] {{
              ^entry(x: {ty}):
                one = builtin.constant <{one}> : {ty};
                rem = math.{rem} (x, one) [] []: <({ty}, {ty}) -> ({ty})>;
                branch.return rem
            }}
        "#
        );
        let ctx = &mut Context::new();
        let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, &input).expect_ok(ctx);
        verify_operation(op, ctx)?;
        let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
        let ret = entry.deref(ctx).get_terminator(ctx).unwrap();
        let original = ret.deref(ctx).get_operand(0);

        // Simplify can only forward values, so it must not replace rem with x.
        assert_eq!(
            apply_match_rewrite(ctx, &mut SimplifyOps, RewriterOrder::default(), op)?,
            IRStatus::Unchanged,
            "{rem}"
        );
        sccp(op, ctx)?;
        verify_operation(op, ctx)?;
        let result = ret.deref(ctx).get_operand(0);
        if let Some(expected) = expected {
            let constant = result
                .defining_op()
                .unwrap()
                .as_op::<ConstantOp>(ctx)
                .unwrap_or_else(|| panic!("{rem} did not fold: {}", op.dyn_op(ctx).disp(ctx)));
            let attr = constant.get_value(ctx);
            assert_eq!(
                attr_cast::<dyn ConstantAttr>(&*attr)
                    .unwrap()
                    .as_const_val(ctx),
                expected,
                "{rem}"
            );
        } else {
            assert_eq!(result, original, "floating remainder depends on x");
        }
    }
    Ok(())
}

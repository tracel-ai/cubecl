//! Tests for the quotient re-indexing of divisibility-guarded range loops.

use cubecl_ir::{AddressType, ContextExt, OpInserter, init_dummy_state};
use cubecl_ir::{
    attributes::IndexAttr,
    dialect::{
        branch::{IfOp, RangeLoopOp},
        math::UDivOp,
    },
    prelude::{Inserter as _, *},
};
use cubecl_opt::passes::quotient_range::QuotientRange;
use cubecl_opt::passes::{mem2reg::Mem2RegPass, sccp::sccp};
use pliron::{
    attribute::Attribute,
    builtin::ops::{ConstantOp, FuncOp},
    context::Context,
    init_env_logger_for_tests,
    irbuild::{
        IRStatus,
        match_rewrite::{RewriterOrder, apply_match_rewrite},
    },
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    opts::dce::dce,
    parsable::parse_from_str,
    result::{ExpectOk, Result},
};

const STORE: &str =
    "memory.store (out, i) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()>;";

fn nop(name: &str) -> String {
    format!(
        "{name} = math.i_add (zero, zero) [] []: <(cube.index , cube.index ) -> (cube.index )>;"
    )
}

/// One fixture for guarded and unguarded loops; pure else-arm nops anchor mutations.
fn loop_ir([lo, hi, base, s, d, step]: [usize; 6], reachability: bool) -> String {
    let gate_nop = if reachability {
        nop("gate_nop")
    } else {
        String::new()
    };
    let gate = format!(
        r#"
        num = math.i_sub (base, prod) [] []: <(cube.index , cube.index ) -> (cube.index )>;
        rem = math.u_rem (num, d) [] []: <(cube.index , cube.index ) -> (cube.index )>;
        hit = cmp.i_equal (rem, zero) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
        branch.if hit then {{
          ^then():
            {STORE}
            branch.yield ()
        }} else {{
          ^else():
            {gate_nop}
            branch.yield ()
        }};
    "#
    );
    let body = if reachability {
        let reach_nop = nop("reach_nop");
        format!(
            r#"
            reach = cmp.u_greater_than_or_equal (base, prod) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            branch.if reach then {{
              ^then():
                {gate}
                branch.yield ()
            }} else {{
              ^else():
                {reach_nop}
                branch.yield ()
            }};
        "#
        )
    } else {
        gate
    };
    format!(
        r#"
    builtin.func @f: builtin.function <(cube.ptr <cube.index , Global<0>>) -> ()> [] {{
      ^entry(out: cube.ptr <cube.index , Global<0>>):
        lo = builtin.constant <cube.index {lo}> : cube.index;
        hi = builtin.constant <cube.index {hi}> : cube.index;
        step = builtin.constant <cube.index {step}> : cube.index;
        base = builtin.constant <cube.index {base}> : cube.index;
        s = builtin.constant <cube.index {s}> : cube.index;
        d = builtin.constant <cube.index {d}> : cube.index;
        zero = builtin.constant <cube.index 0> : cube.index;
        branch.range_loop (lo, hi, step) [] []: <(cube.index , cube.index , cube.index ) -> ()> {{
          ^body(i: cube.index ):
            prod = math.i_mul (i, s) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            {body}
            branch.yield ()
        }};
        branch.return
    }}
    "#
    )
}

fn guarded_loop(lo: usize, hi: usize, base: usize, s: usize, d: usize, step: usize) -> String {
    loop_ir([lo, hi, base, s, d, step], true)
}

fn test_context(address: AddressType) -> Context {
    init_env_logger_for_tests!();
    let mut ctx = Context::new();
    init_dummy_state(&mut ctx);
    ctx.set_address_type(address);
    ctx
}

fn parse_ir(ctx: &mut Context, input: &str) -> Result<Ptr<Operation>> {
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(op, ctx)?;
    Ok(op)
}

fn rewrite(input: &str) -> Result<IRStatus> {
    let ctx = &mut test_context(AddressType::U32);
    let op = parse_ir(ctx, input)?;

    let mut pass = QuotientRange;
    let changed = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    verify_operation(op, ctx)?;
    assert_eq!(
        apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?,
        IRStatus::Unchanged
    );

    Ok(changed)
}

/// Match the backend cleanup needed by the conditional polyfill's locals.
fn cleanup(ctx: &mut Context, op: Ptr<Operation>) -> Result<()> {
    sccp(op, ctx)?;
    apply_match_rewrite(
        ctx,
        &mut cubecl_ir::rewrite::Canonicalize,
        RewriterOrder::default(),
        op,
    )?;
    Mem2RegPass.run(op, ctx, &mut AnalysisManager::default())?;
    sccp(op, ctx)?;
    apply_match_rewrite(
        ctx,
        &mut cubecl_ir::rewrite::Canonicalize,
        RewriterOrder::default(),
        op,
    )?;
    dce(op, ctx)?;
    verify_operation(op, ctx)
}

#[test]
fn declines_unsupported_windows() -> Result<()> {
    assert_eq!(
        rewrite(&loop_ir([9, 49, 48, 1, 8, 1], false))?,
        IRStatus::Unchanged
    );
    for (case, values) in [
        ("zero divisor", [9, 14, 48, 1, 0, 1]),
        ("unit divisor", [9, 14, 48, 1, 1, 1]),
        ("small divisor", [9, 19, 48, 1, 2, 1]),
        ("non-unit loop step", [9, 49, 48, 1, 8, 2]),
    ] {
        assert_eq!(
            rewrite(&loop_ir(values, true))?,
            IRStatus::Unchanged,
            "{case}"
        );
    }
    Ok(())
}

/// Check the cleaned IR, rather than a snapshot of incidental SSA names.
#[test]
fn quotient_loop_has_five_iterations() -> Result<()> {
    for dilation in [3, 4, 8] {
        let ctx = &mut test_context(AddressType::U32);
        let input = guarded_loop(
            33 - 3 * dilation,
            33 + 2 * dilation,
            32 + 2 * dilation,
            1,
            dilation,
            1,
        );
        let op = parse_ir(ctx, &input)?;
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
        cleanup(ctx, op)?;
        let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
        let loops = entry
            .deref(ctx)
            .iter(ctx)
            .filter_map(|op| op.as_op::<RangeLoopOp>(ctx))
            .collect::<Vec<_>>();
        assert_eq!(loops.len(), 1, "the static fallback must be removed");
        let index = |value: Value| {
            let attr = value
                .defining_op()
                .unwrap()
                .as_op::<ConstantOp>(ctx)
                .unwrap()
                .get_value(ctx);
            (&*attr as &dyn Attribute)
                .downcast_ref::<IndexAttr>()
                .unwrap()
                .0
        };
        assert_eq!(
            (index(loops[0].start(ctx)), index(loops[0].end(ctx))),
            (0, 5)
        );
        // Unit stride needs no division in the per-candidate recovery chain.
        assert!(
            !loops[0]
                .loop_body(ctx)
                .deref(ctx)
                .iter(ctx)
                .any(|op| op.as_op::<UDivOp>(ctx).is_some())
        );
    }
    Ok(())
}

#[test]
fn declines_when_effects_escape_or_the_divisor_varies() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1);
    for (case, from, to) in [
        ("effect outside guards", "prod = math.i_mul".into(), format!("{STORE}\n prod = math.i_mul")),
        ("effect in divisibility else", nop("gate_nop"), STORE.into()),
        ("effect in reachability else", nop("reach_nop"), STORE.into()),
        ("varying divisor", "rem = math.u_rem (num, d)".into(),
         "inner_d = math.i_add (d, i) [] []: <(cube.index , cube.index ) -> (cube.index )>;\n rem = math.u_rem (num, inner_d)".into()),
    ] {
        assert_eq!(input.matches(&from).count(), 1, "{case}: missing fixture anchor");
        assert_eq!(rewrite(&input.replace(&from, &to))?, IRStatus::Unchanged, "{case}");
    }
    Ok(())
}

#[test]
fn preserves_return_inside_the_gate() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1).replacen("branch.yield ()", "branch.return", 1);
    let ctx = &mut test_context(AddressType::U32);
    let op = parse_ir(ctx, &input)?;
    apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
    cleanup(ctx, op)?;
    let after = op.dyn_op(ctx).disp(ctx).to_string();
    // The optimized body's early return plus the final function return.
    assert_eq!(after.matches("branch.return").count(), 2);
    Ok(())
}

#[test]
fn preserves_remainder_used_by_the_body() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1)
        .replace("zero = builtin.constant <cube.index 0> : cube.index;", "")
        .replace("num = math.i_sub", "zero = builtin.constant <cube.index 0> : cube.index;\n                num = math.i_sub")
        .replace("memory.store (out, i)", "memory.store (out, rem)")
        .replace("reach_nop = math.i_add (zero, zero)", "reach_nop = math.i_add (lo, lo)");
    // Exercise cloning through an additional nested guard as well as a local zero.
    let input = input.replace("memory.store (out, rem) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()>;",
        "branch.if reach then { ^nested(): memory.store (out, rem) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()>; branch.yield () } else { ^else(): branch.yield () };");
    assert_eq!(rewrite(&input)?, IRStatus::Changed);
    Ok(())
}

#[test]
fn declines_when_return_escapes_the_gate() -> Result<()> {
    for name in ["gate_nop", "reach_nop"] {
        let mut input = guarded_loop(9, 49, 48, 1, 8, 1);
        let start = input.find(&nop(name)).expect("missing else-arm nop");
        let end = start + input[start..].find("branch.yield ()").unwrap() + "branch.yield ()".len();
        input.replace_range(start..end, "branch.return");
        assert_eq!(input.matches("branch.return").count(), 2);
        assert_eq!(rewrite(&input)?, IRStatus::Unchanged, "{name}");
    }
    Ok(())
}

fn dynamic_loop() -> String {
    let names = ["lo", "hi", "base", "s", "d"];
    let mut input = guarded_loop(0, 0, 0, 0, 0, 1)
        .replace("Global<0>>) ->", "Global<0>>, cube.index, cube.index, cube.index, cube.index, cube.index) ->")
        .replace("Global<0>>):", "Global<0>>, lo: cube.index, hi: cube.index, base: cube.index, s: cube.index, d: cube.index):");
    for name in names {
        input = input.replace(
            &format!("{name} = builtin.constant <cube.index 0> : cube.index;"),
            "",
        );
    }
    input
}

/// Specialize arguments only *after* rewriting, so this checks the emitted
/// runtime safety and profitability condition with both index widths.
fn runtime_uses_quotient(args: [usize; 5], address: AddressType) -> Result<bool> {
    let input = dynamic_loop();
    let ctx = &mut test_context(address);
    let op = parse_ir(ctx, &input)?;
    let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
    let original = entry
        .deref(ctx)
        .iter(ctx)
        .find(|op| op.as_op::<RangeLoopOp>(ctx).is_some())
        .unwrap();
    assert_eq!(
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?,
        IRStatus::Changed
    );
    verify_operation(op, ctx)?;
    let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
    let mut inserter = OpInserter::new_at_block_start(entry);
    for (index, value) in args.into_iter().enumerate() {
        let constant = ConstantOp::new(ctx, Box::new(IndexAttr::new(value)));
        inserter.append_op(ctx, &constant);
        let argument = entry.deref(ctx).get_argument(index + 1);
        argument.replace_all_uses_with(ctx, &constant.get_result(ctx));
    }
    cleanup(ctx, op)?;
    let loops = entry
        .deref(ctx)
        .iter(ctx)
        .filter(|op| op.as_op::<RangeLoopOp>(ctx).is_some())
        .collect::<Vec<_>>();
    assert_eq!(loops.len(), 1);
    Ok(loops[0] != original)
}

#[test]
fn runtime_checks_safety_and_profitability() -> Result<()> {
    for (args, expected) in [
        ([9, 49, 48, 1, 8], true),
        ([9, 19, 48, 1, 0], false),
        ([9, 19, 48, 1, 1], false),
        ([9, 19, 48, 1, 2], false),
        ([9, 9, 48, 1, 8], false),
        ([8, 9, 48, 1, 4], false),
        ([0, 4, 100, 100, 4], false),
        ([0, 8, u32::MAX as usize, 1, 1], false),
        ([0, 8, 48, 0, 8], false),
        ([9, 0, 48, 1, 8], false),
        ([1 << 31, (1 << 31) + 8, 16, 2, 8], false),
        ([(1 << 31) - 8, 1 << 31, u32::MAX as usize, 2, 8], false),
    ] {
        assert_eq!(
            runtime_uses_quotient(args, AddressType::U32)?,
            expected,
            "{args:?}"
        );
    }
    #[cfg(target_pointer_width = "64")]
    {
        assert!(runtime_uses_quotient(
            [1 << 31, (1 << 31) + 8, 16, 2, 8],
            AddressType::U64
        )?);
        assert!(!runtime_uses_quotient(
            [1 << 63, (1 << 63) + 8, 16, 2, 8],
            AddressType::U64
        )?);
    }
    Ok(())
}

/// Traverse nested regions without relying on incidental SSA names.
fn descendants(ctx: &Context, op: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut result = vec![op];
    for region in op.regions(ctx) {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                result.extend(descendants(ctx, child));
            }
        }
    }
    result
}

fn wrap_loop(input: &str, prefix: &str, suffix: &str) -> String {
    let start = input
        .find("cond =")
        .unwrap_or_else(|| input.find("branch.range_loop").unwrap());
    let end = input.rfind("branch.return").unwrap();
    format!(
        "{}{}{}{}{}",
        &input[..start],
        prefix,
        &input[start..end],
        suffix,
        &input[end..]
    )
}

const OUTER_PREFIX: &str = "branch.range_loop (lo, hi, step) [] []: <(cube.index, cube.index, cube.index) -> ()> { ^outer(j: cube.index):";
const OUTER_SUFFIX: &str = "branch.yield () };";

#[test]
fn hoists_preparation_across_invariant_outer_loops() -> Result<()> {
    for depth in [1, 2] {
        let mut input = dynamic_loop();
        for level in 0..depth {
            input = wrap_loop(
                &input,
                &OUTER_PREFIX.replace("j:", &format!("j{level}:")),
                OUTER_SUFFIX,
            );
        }
        let ctx = &mut test_context(AddressType::U32);
        let op = parse_ir(ctx, &input)?;
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
        verify_operation(op, ctx)?;
        let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
        let ops = entry.deref(ctx).iter(ctx).collect::<Vec<_>>();
        let prep = ops
            .iter()
            .position(|op| op.as_op::<IfOp>(ctx).is_some())
            .unwrap();
        let outer = ops
            .iter()
            .position(|op| op.as_op::<RangeLoopOp>(ctx).is_some())
            .unwrap();
        assert!(prep < outer);
        let prep_ops = descendants(ctx, ops[prep]);
        assert_eq!(
            prep_ops
                .iter()
                .filter(|op| op.as_op::<UDivOp>(ctx).is_some())
                .count(),
            3
        );
        // No quotient-bound divisions remain inside either outer loop. The
        // dynamic-stride candidate recovery has its own division, as expected.
        assert_eq!(
            descendants(ctx, ops[outer])
                .iter()
                .filter(|op| op.as_op::<UDivOp>(ctx).is_some())
                .count(),
            1
        );
        assert_eq!(
            apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?,
            IRStatus::Unchanged
        );
    }
    Ok(())
}

#[test]
fn keeps_preparation_with_varying_inputs_and_inside_conditions() -> Result<()> {
    for (from, to) in [
        ("(lo, hi, step)", "(j, hi, step)"),
        ("(lo, hi, step)", "(lo, j, step)"),
        ("(base, prod)", "(j, prod)"),
        ("(i, s)", "(i, j)"),
        ("(num, d)", "(num, j)"),
    ] {
        let input = dynamic_loop().replace(from, to);
        let input = wrap_loop(&input, OUTER_PREFIX, OUTER_SUFFIX);
        let ctx = &mut test_context(AddressType::U32);
        let op = parse_ir(ctx, &input)?;
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
        verify_operation(op, ctx)?;
        let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
        assert!(
            !entry
                .deref(ctx)
                .iter(ctx)
                .any(|op| op.as_op::<IfOp>(ctx).is_some()),
            "{from}"
        );
        let outer = entry
            .deref(ctx)
            .iter(ctx)
            .find_map(|op| op.as_op::<RangeLoopOp>(ctx))
            .unwrap();
        assert_eq!(
            outer
                .loop_body(ctx)
                .deref(ctx)
                .iter(ctx)
                .filter(|op| op.as_op::<IfOp>(ctx).is_some())
                .count(),
            2,
            "preparation and dispatch must remain inside: {from}"
        );
    }
    let input = wrap_loop(
        &dynamic_loop(),
        "cond = cmp.i_equal (lo, hi) [] []: <(cube.index, cube.index) -> (cube.bool)>; branch.if cond then { ^conditional():",
        "branch.yield () } else { ^else(): branch.yield () };",
    );
    let input = wrap_loop(&input, OUTER_PREFIX, OUTER_SUFFIX);
    let ctx = &mut test_context(AddressType::U32);
    let op = parse_ir(ctx, &input)?;
    // Find the existing condition before rewriting. Preparation stays inside it.
    let conditional = descendants(ctx, op)
        .into_iter()
        .find_map(|op| op.as_op::<IfOp>(ctx))
        .unwrap();
    apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
    verify_operation(op, ctx)?;
    assert_eq!(
        conditional
            .then_block(ctx)
            .deref(ctx)
            .iter(ctx)
            .filter(|op| op.as_op::<IfOp>(ctx).is_some())
            .count(),
        2
    );
    Ok(())
}

#[test]
fn dynamic_small_divisors_eliminate_all_preparation() -> Result<()> {
    // Zero is covered by runtime_checks_safety_and_profitability; folding the
    // source loop's remainder by zero is not a defined arithmetic operation.
    for d in 1..=2 {
        let ctx = &mut test_context(AddressType::U32);
        let op = parse_ir(ctx, &dynamic_loop())?;
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?;
        let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
        let mut inserter = OpInserter::new_at_block_start(entry);
        let constant = ConstantOp::new(ctx, Box::new(IndexAttr::new(d)));
        inserter.append_op(ctx, &constant);
        let divisor = entry.deref(ctx).get_argument(5);
        divisor.replace_all_uses_with(ctx, &constant.get_result(ctx));
        cleanup(ctx, op)?;
        let ops = descendants(ctx, op);
        assert_eq!(
            ops.iter()
                .filter(|op| op.as_op::<RangeLoopOp>(ctx).is_some())
                .count(),
            1
        );
        assert!(
            !entry
                .deref(ctx)
                .iter(ctx)
                .any(|op| op.as_op::<IfOp>(ctx).is_some())
        );
        assert!(!ops.iter().any(|op| op.as_op::<UDivOp>(ctx).is_some()));
        let after = op.dyn_op(ctx).disp(ctx).to_string();
        assert!(!after.contains("bitwise.shr"));
        assert!(!after.contains("memory.declare_variable"));
    }
    Ok(())
}

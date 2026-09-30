//! Tests for the quotient re-indexing of divisibility-guarded range loops.

use cubecl_ir::{AddressType, ContextExt, OpInserter, init_dummy_state};
use cubecl_ir::{
    attributes::{BoolAttr, IndexAttr},
    dialect::branch::IfOp,
    prelude::{Inserter as _, *},
};
use cubecl_opt::passes::quotient_range::QuotientRange;
use cubecl_opt::passes::sccp::sccp;
use expect_test::expect;
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

fn rewrite(input: &str) -> Result<(IRStatus, String)> {
    let ctx = &mut test_context(AddressType::U32);
    let op = parse_ir(ctx, input)?;

    let mut pass = QuotientRange;
    let changed = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    verify_operation(op, ctx)?;

    Ok((
        changed,
        Operation::get_op_dyn(op, ctx)
            .disp(ctx)
            .to_string()
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n"),
    ))
}

/// Dilation only moves the bound, never the trip count. The profitability
/// check wants at least as many skipped iterations as kept ones.
#[test]
fn trip_count_is_independent_of_the_divisor() -> Result<()> {
    for dilation in [3usize, 4, 8] {
        let hi = 9 + 5 * dilation;
        let (changed, _) = rewrite(&guarded_loop(9, hi, 48, 1, dilation, 1))?;
        assert_eq!(
            changed,
            IRStatus::Changed,
            "dilation {dilation} should be re-indexed"
        );
    }
    Ok(())
}

#[test]
fn declines_unsupported_windows() -> Result<()> {
    for (case, values) in [
        ("unprofitable window", [8, 9, 48, 1, 4, 1]),
        ("zero divisor", [9, 14, 48, 1, 0, 1]),
        ("unit divisor", [9, 14, 48, 1, 1, 1]),
        ("small divisor", [9, 19, 48, 1, 2, 1]),
        ("non-unit loop step", [9, 49, 48, 1, 8, 2]),
        ("product overflow", [1 << 31, (1 << 31) + 8, 16, 2, 8, 1]),
        ("endpoint overflow", [(1 << 31) - 8, 1 << 31, 16, 2, 8, 1]),
    ] {
        assert_eq!(
            rewrite(&loop_ir(values, true))?.0,
            IRStatus::Unchanged,
            "{case}"
        );
    }
    Ok(())
}

/// Without the reachability guard, an `i * s` past `base` wraps the numerator
/// onto a quotient the rewrite cannot account for — those iterations may reach
/// the body in the source loop, and the rewrite would drop them.
#[test]
fn declines_without_the_reachability_guard() -> Result<()> {
    let (changed, _) = rewrite(&loop_ir([9, 49, 48, 1, 8, 1], false))?;
    assert_eq!(changed, IRStatus::Unchanged);
    Ok(())
}

/// A same-size conv-transpose with `kernel = 5` and `dilation = 8`, at the
/// interior output `out_y = 32`: the window is `y_start = 9 .. y_end = 49` and
/// the numerator is `48 - in_y`. The source loop runs 40 times for 5 taps.
///
/// After the rewrite the quotient bounds are `(48 -| 49) / 8 = 0` and
/// `(48 -| 9) / 8 + 1 = 5`, so the loop runs once per tap. Counting back down
/// from `q_last = 4`, `in_y` comes back as `(48 - (4 - q) * 8) / 1`, walking
/// `16, 24, 32, 40, 48` the way the source loop did.
#[test]
fn rewrites_the_dilated_window_loop() -> Result<()> {
    let (changed, after) = rewrite(&guarded_loop(9, 49, 48, 1, 8, 1))?;
    assert_eq!(changed, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.ptr <cube.index , Global<0>>) -> ()>
        {
          ^entry_block3v1(out_v0: cube.ptr <cube.index , Global<0>>) !0:
            lo_v1 = builtin.constant <cube.index 9> : cube.index  !1;
            hi_v2 = builtin.constant <cube.index 49> : cube.index  !2;
            step_v3 = builtin.constant <cube.index 1> : cube.index  !3;
            base_v4 = builtin.constant <cube.index 48> : cube.index  !4;
            s_v5 = builtin.constant <cube.index 1> : cube.index  !5;
            d_v6 = builtin.constant <cube.index 8> : cube.index  !6;
            zero_v7 = builtin.constant <cube.index 0> : cube.index  !7;
            v16 = builtin.constant <cube.index 1> : cube.index ;
            v17 = cmp.u_max (s_v5, v16) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v18 = builtin.constant <cube.index 3> : cube.index ;
            v19 = cmp.u_max (d_v6, v18) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v20 = math.i_mul (lo_v1, v17) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v21 = cmp.u_min (v20, base_v4) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v22 = math.i_sub (base_v4, v21) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v23 = math.i_mul (hi_v2, v17) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v24 = cmp.u_min (v23, base_v4) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v25 = math.i_sub (base_v4, v24) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v26 = cmp.u_greater_than_or_equal (base_v4, v23) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v27 = cube.cast (v26) [] []: <(cube.bool ) -> (cube.index )>;
            v28 = math.u_div (v25, v19) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v29 = math.i_add (v28, v27) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v30 = math.u_div (v22, v19) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v31 = builtin.constant <cube.index 1> : cube.index ;
            v32 = math.i_add (v30, v31) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v33 = math.i_sub (v32, v29) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v34 = math.i_sub (hi_v2, lo_v1) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v35 = builtin.constant <cube.index 1> : cube.index ;
            v36 = cmp.u_max (s_v5, v35) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v37 = builtin.constant <cube.index 4294967295> : cube.index ;
            v38 = math.u_div (v37, v36) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v39 = cmp.u_less_than_or_equal (hi_v2, v38) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v40 = builtin.constant <cube.index 1> : cube.index ;
            v41 = cmp.u_greater_than_or_equal (s_v5, v40) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v42 = builtin.constant <cube.index 3> : cube.index ;
            v43 = cmp.u_greater_than_or_equal (d_v6, v42) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v44 = cube.bool_and (v41, v43) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
            v45 = cmp.u_less_than (lo_v1, hi_v2) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v46 = cube.bool_and (v44, v45) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
            v47 = cube.bool_and (v46, v39) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
            v48 = builtin.constant <cube.index 2> : cube.index ;
            v49 = math.u_div (v34, v48) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v50 = cmp.u_less_than_or_equal (v33, v49) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v51 = cube.bool_and (v47, v50) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
            branch.if v51 then
            {
              ^then_block9v1():
                branch.range_loop (v29, v32, step_v3) [] [quotient_range_unswitched: builtin.unit ]: <(cube.index , cube.index , cube.index ) -> ()>
                {
                  ^body_block11v1(v52: cube.index ):
                    v53 = math.i_sub (v52, v29) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                    v54 = math.i_sub (v30, v53) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                    v55 = math.i_mul (v54, d_v6) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                    v56 = math.i_sub (base_v4, v55) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                    v57 = builtin.constant <cube.bool true> : cube.bool ;
                    v58 = cmp.u_greater_than_or_equal (v56, lo_v1) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                    v59 = cmp.u_less_than (v56, hi_v2) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                    v60 = cube.bool_and (v58, v59) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                    v61 = cube.bool_and (v57, v60) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                    branch.if v61 then
                    {
                      ^then_block12v1():
                        memory.store (out_v0, v56) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()> !8;
                        branch.yield ()
                    } else
                    {
                      ^else_block13v1():
                        branch.yield ()
                    };
                    branch.yield ()
                };
                branch.yield ()
            } else
            {
              ^else_block10v1():
                branch.range_loop (lo_v1, hi_v2, step_v3) [] [quotient_range_unswitched: builtin.unit ]: <(cube.index , cube.index , cube.index ) -> ()>
                {
                  ^body_block4v1(i_v8: cube.index ) !9:
                    prod_v9 = math.i_mul (i_v8, s_v5) [] []: <(cube.index , cube.index ) -> (cube.index )> !10;
                    reach_v10 = cmp.u_greater_than_or_equal (base_v4, prod_v9) [] []: <(cube.index , cube.index ) -> (cube.bool )> !11;
                    branch.if reach_v10 then
                    {
                      ^then_block5v1() !12:
                        num_v11 = math.i_sub (base_v4, prod_v9) [] []: <(cube.index , cube.index ) -> (cube.index )> !13;
                        rem_v12 = math.u_rem (num_v11, d_v6) [] []: <(cube.index , cube.index ) -> (cube.index )> !14;
                        hit_v13 = cmp.i_equal (rem_v12, zero_v7) [] []: <(cube.index , cube.index ) -> (cube.bool )> !15;
                        branch.if hit_v13 then
                        {
                          ^then_block6v1() !16:
                            memory.store (out_v0, i_v8) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()> !17;
                            branch.yield () !18
                        } else
                        {
                          ^else_block7v1() !19:
                            gate_nop_v14 = math.i_add (zero_v7, zero_v7) [] []: <(cube.index , cube.index ) -> (cube.index )> !20;
                            branch.yield () !21
                        } !22;
                        branch.yield () !23
                    } else
                    {
                      ^else_block8v1() !24:
                        reach_nop_v15 = math.i_add (zero_v7, zero_v7) [] []: <(cube.index , cube.index ) -> (cube.index )> !25;
                        branch.yield () !26
                    } !27;
                    branch.yield () !28
                } !29;
                branch.yield ()
            };
            branch.return  !30
        }"#]].assert_eq(&after);
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
        assert_eq!(rewrite(&input.replace(&from, &to))?.0, IRStatus::Unchanged, "{case}");
    }
    Ok(())
}

/// Which `i` the source loop runs its body on, in order. `d == 0` is left to
/// the caller: `u_rem` by zero says nothing.
fn source_iterations(lo: u32, hi: u32, base: u32, s: u32, d: u32) -> Vec<u32> {
    (lo..hi)
        .filter(|i| base >= i.wrapping_mul(s) && base.wrapping_sub(i.wrapping_mul(s)) % d == 0)
        .collect()
}

/// The same, for the loop the pass emits. This mirrors the `quotient_bounds`,
/// `quotient_applicable` and `recover_tap` polyfills op for op, down to the
/// unswitch that parks losing windows in the source arm, and has to be
/// updated with them; the golden IR snapshot is what ties the two together.
fn rewritten_iterations(lo: u32, hi: u32, base: u32, s: u32, d: u32) -> Vec<u32> {
    // mirrors quotient_bounds (clamped divisors) and quotient_applicable
    let s_safe = s.max(1);
    let d_safe = d.max(3);
    let saturating_sub = |bound: u32| base.wrapping_sub(base.min(bound.wrapping_mul(s_safe)));
    let q_start = saturating_sub(hi) / d_safe + u32::from(base >= hi.wrapping_mul(s_safe));
    let q_last = saturating_sub(lo) / d_safe;
    let q_end = q_last.wrapping_add(1);
    let taps = q_end.wrapping_sub(q_start);
    let trips = hi.wrapping_sub(lo);
    if !(s >= 1 && d >= 3 && lo < hi && hi <= u32::MAX / s_safe && taps <= trips / 2) {
        return source_iterations(lo, hi, base, s, d);
    }

    // mirrors recover_tap
    (q_start..q_end)
        .filter_map(|q| {
            let tap = q_last - (q - q_start);
            let numerator = tap.wrapping_mul(d);
            let scaled = base.wrapping_sub(numerator);
            let recovered = scaled / s;
            let exact = scaled % s == 0;
            let in_range = recovered >= lo && recovered < hi;
            (exact && in_range).then_some(recovered)
        })
        .collect()
}

/// The rewrite has to run the body on the same iterations, in the same order.
/// Order matters because the quotient climbs as `i` falls, and a body whose
/// last write wins would see the difference.
#[test]
fn preserves_the_source_iterations() {
    for lo in 0u32..8 {
        for hi in lo..34 {
            for base in [0u32, 1, 5, 7, 16, 31, 32, 47, 48, 49, 64, 100] {
                for s in 0u32..6 {
                    for d in 1u32..10 {
                        let want = source_iterations(lo, hi, base, s, d);
                        let got = rewritten_iterations(lo, hi, base, s, d);
                        assert_eq!(got, want, "lo={lo} hi={hi} base={base} s={s} d={d}");
                    }
                }
            }
        }
    }
}

/// A zero `s` or a unit `d` has no holes to skip, so the unswitch parks those
/// operands in the source arm and the loop walks the range it always did.
#[test]
fn small_operands_walk_the_source_range() {
    for lo in 0u32..6 {
        for hi in lo..12 {
            for base in [0u32, 5, 48] {
                for s in 0u32..4 {
                    for d in [1u32, 2, 8] {
                        if s >= 1 && d >= 2 {
                            continue;
                        }
                        assert_eq!(
                            rewritten_iterations(lo, hi, base, s, d),
                            source_iterations(lo, hi, base, s, d),
                            "lo={lo} hi={hi} base={base} s={s} d={d}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn preserves_boundary_iterations() {
    for (lo, hi, base, s, d) in [
        (1 << 31, (1 << 31) + 8, 16, 2, 8),
        ((1 << 31) - 8, 1 << 31, u32::MAX, 2, 8),
        (u32::MAX - 8, u32::MAX, u32::MAX, 1, 8),
        (0, 8, u32::MAX, 1, 1),
        (9, 0, u32::MAX, 2, 8),
    ] {
        assert_eq!(
            rewritten_iterations(lo, hi, base, s, d),
            source_iterations(lo, hi, base, s, d),
            "lo={lo} hi={hi} base={base} s={s} d={d}"
        );
    }
}

#[test]
fn preserves_return_inside_the_gate() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1).replacen("branch.yield ()", "branch.return", 1);
    assert_eq!(input.matches("branch.return").count(), 2);
    let (changed, after) = rewrite(&input)?;
    assert_eq!(changed, IRStatus::Changed);
    // Function exit plus one return in each arm of the unswitch.
    assert_eq!(after.matches("branch.return").count(), 3);
    Ok(())
}

#[test]
fn preserves_remainder_used_by_the_body() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1)
        .replace("zero = builtin.constant <cube.index 0> : cube.index;", "")
        .replace("num = math.i_sub", "zero = builtin.constant <cube.index 0> : cube.index;\n                num = math.i_sub")
        .replace("memory.store (out, i)", "memory.store (out, rem)")
        .replace("reach_nop = math.i_add (zero, zero)", "reach_nop = math.i_add (lo, lo)");
    assert_eq!(rewrite(&input)?.0, IRStatus::Changed);
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
        assert_eq!(rewrite(&input)?.0, IRStatus::Unchanged, "{name}");
    }
    Ok(())
}

/// Specialize arguments only *after* rewriting, so this checks the emitted
/// runtime condition rather than the separate compile-time decline logic.
fn runtime_uses_quotient(args: [usize; 5], address: AddressType) -> Result<bool> {
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
    let ctx = &mut test_context(address);
    let op = parse_ir(ctx, &input)?;
    assert_eq!(
        apply_match_rewrite(ctx, &mut QuotientRange, RewriterOrder::default(), op)?,
        IRStatus::Changed
    );
    verify_operation(op, ctx)?;
    let entry = op.as_op::<FuncOp>(ctx).unwrap().get_entry_block(ctx);
    let unswitch = entry
        .deref(ctx)
        .iter(ctx)
        .find_map(|op| op.as_op::<IfOp>(ctx))
        .unwrap();
    let mut inserter = OpInserter::new_at_block_start(entry);
    for (index, value) in args.into_iter().enumerate() {
        let constant = ConstantOp::new(ctx, Box::new(IndexAttr::new(value)));
        inserter.append_op(ctx, &constant);
        let argument = entry.deref(ctx).get_argument(index + 1);
        argument.replace_all_uses_with(ctx, &constant.get_result(ctx));
    }
    sccp(op, ctx)?;
    verify_operation(op, ctx)?;
    let constant = unswitch
        .condition(ctx)
        .defining_op()
        .unwrap()
        .as_op::<ConstantOp>(ctx)
        .unwrap();
    let attr = constant.get_value(ctx);
    Ok((&*attr as &dyn Attribute)
        .downcast_ref::<BoolAttr>()
        .unwrap()
        .0)
}

#[test]
fn runtime_checks_safety_and_profitability() -> Result<()> {
    for (args, expected) in [
        ([9, 49, 48, 1, 8], true),
        ([9, 19, 48, 1, 2], false),
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

/// The rewritten loop must not look like a fresh match, or the pass would
/// unswitch its own unswitch forever.
#[test]
fn is_idempotent() -> Result<()> {
    let ctx = &mut test_context(AddressType::U32);
    let input = guarded_loop(9, 49, 48, 1, 8, 1);
    let op = parse_ir(ctx, &input)?;

    let mut pass = QuotientRange;
    let first = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    assert_eq!(first, IRStatus::Changed);
    let once = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();

    let again = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    assert_eq!(again, IRStatus::Unchanged, "pass matched its own output");
    let twice = Operation::get_op_dyn(op, ctx).disp(ctx).to_string();
    assert_eq!(once, twice);
    Ok(())
}

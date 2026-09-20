//! Tests for the quotient re-indexing of divisibility-guarded range loops.

use cubecl_opt::passes::quotient_range::QuotientRange;
use expect_test::expect;
use pliron::{
    context::Context,
    init_env_logger_for_tests,
    irbuild::{
        IRStatus,
        match_rewrite::{RewriterOrder, apply_match_rewrite},
    },
    irfmt::parsers::spaced,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    result::{ExpectOk, Result},
};

/// `for i in lo..hi { if (base - i * s) % d == 0 { out[0] = i } }`, the shape
/// `conv_transpose2d_direct_kernel` walks its dilated window with.
fn guarded_loop(lo: usize, hi: usize, base: usize, s: usize, d: usize, step: usize) -> String {
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
            num = math.i_sub (base, prod) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            rem = math.u_rem (num, d) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            hit = cmp.i_equal (rem, zero) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            branch.if hit then {{
              ^then():
                memory.store (out, i) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()>;
                branch.yield ()
            }} else {{
              ^else():
                branch.yield ()
            }};
            branch.yield ()
        }};
        branch.return
    }}
  "#
    )
}

fn rewrite(input: &str) -> Result<(IRStatus, String)> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(op, ctx)?;

    let mut pass = QuotientRange;
    let changed = apply_match_rewrite(ctx, &mut pass, RewriterOrder::default(), op)?;
    verify_operation(op, ctx)?;

    Ok((
        changed,
        Operation::get_op_dyn(op, ctx).disp(ctx).to_string(),
    ))
}

/// Dilation only moves the bound, never the trip count.
#[test]
fn trip_count_is_independent_of_the_divisor() -> Result<()> {
    for dilation in [1usize, 2, 3, 4, 8] {
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

/// A same-size conv-transpose with `kernel = 5` and `dilation = 8`, at the
/// interior output `out_y = 32`: the window is `y_start = 9 .. y_end = 49` and
/// the numerator is `48 - in_y`. The source loop runs 40 times for 5 taps.
///
/// After the rewrite the bounds are `(48 -| 49) / 8 = 0` and
/// `(48 -| 9) / 8 + 1 = 5`, so the loop runs once per tap, and `in_y` is
/// recovered as `(48 - q * 8) / 1`.
#[test]
fn rewrites_the_dilated_window_loop() -> Result<()> {
    let (changed, after) = rewrite(&guarded_loop(9, 49, 48, 1, 8, 1))?;
    assert_eq!(changed, IRStatus::Changed);
    expect![[r#"
        builtin.func @f: builtin.function <(cube.ptr <cube.index , Global<0>>) -> ()> 
        {
          ^entry_block1v1(out_v0: cube.ptr <cube.index , Global<0>>) !0:
            lo_v1 = builtin.constant <cube.index 9> : cube.index  !1;
            hi_v2 = builtin.constant <cube.index 49> : cube.index  !2;
            step_v3 = builtin.constant <cube.index 1> : cube.index  !3;
            base_v4 = builtin.constant <cube.index 48> : cube.index  !4;
            s_v5 = builtin.constant <cube.index 1> : cube.index  !5;
            d_v6 = builtin.constant <cube.index 8> : cube.index  !6;
            zero_v7 = builtin.constant <cube.index 0> : cube.index  !7;
            v13 = builtin.constant <cube.index 0> : cube.index ;
            v14 = builtin.constant <cube.index 1> : cube.index ;
            v15 = cmp.i_equal (s_v5, v13) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v16 = cmp.i_equal (d_v6, v13) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
            v17 = cube.bool_or (v15, v16) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
            v18 = cmp.u_max (d_v6, v14) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v19 = cmp.u_max (s_v5, v14) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v20 = math.i_mul (lo_v1, s_v5) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v21 = cmp.u_min (base_v4, v20) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v22 = math.i_sub (base_v4, v21) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v23 = math.i_mul (hi_v2, s_v5) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v24 = cmp.u_min (base_v4, v23) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v25 = math.i_sub (base_v4, v24) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v26 = math.u_div (v25, v18) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v27 = math.u_div (v22, v18) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v28 = math.i_add (v27, v14) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            v29 = cube.select (v17, lo_v1, v26) [] []: <(cube.bool , cube.index , cube.index ) -> (cube.index )>;
            v30 = cube.select (v17, hi_v2, v28) [] []: <(cube.bool , cube.index , cube.index ) -> (cube.index )>;
            branch.range_loop (v29, v30, v14) [] []: <(cube.index , cube.index , cube.index ) -> ()>
            {
              ^body_block5v1(v31: cube.index ):
                v32 = math.i_mul (v31, d_v6) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                v33 = cmp.u_greater_than_or_equal (base_v4, v32) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                v34 = math.i_sub (base_v4, v32) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                v35 = math.u_rem (v34, v19) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                v36 = cmp.i_equal (v35, v13) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                v37 = math.u_div (v34, v19) [] []: <(cube.index , cube.index ) -> (cube.index )>;
                v38 = cmp.u_greater_than_or_equal (v37, lo_v1) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                v39 = cmp.u_less_than (v37, hi_v2) [] []: <(cube.index , cube.index ) -> (cube.bool )>;
                v40 = cube.bool_and (v38, v39) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                v41 = cube.bool_and (v33, v36) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                v42 = cube.bool_and (v41, v40) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                v43 = cube.bool_or (v17, v42) [] []: <(cube.bool , cube.bool ) -> (cube.bool )>;
                i_v44 = cube.select (v17, v31, v37) [] []: <(cube.bool , cube.index , cube.index ) -> (cube.index )> !8;
                branch.if v43 then 
                {
                  ^then_block6v1():
                    prod_v9 = math.i_mul (i_v44, s_v5) [] []: <(cube.index , cube.index ) -> (cube.index )> !9;
                    num_v10 = math.i_sub (base_v4, prod_v9) [] []: <(cube.index , cube.index ) -> (cube.index )> !10;
                    rem_v11 = math.u_rem (num_v10, d_v6) [] []: <(cube.index , cube.index ) -> (cube.index )> !11;
                    hit_v12 = cmp.i_equal (rem_v11, zero_v7) [] []: <(cube.index , cube.index ) -> (cube.bool )> !12;
                    branch.if hit_v12 then 
                    {
                      ^then_block3v1() !13:
                        memory.store (out_v0, i_v44) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()> !14;
                        branch.yield () !15
                    } else 
                    {
                      ^else_block4v1() !16:
                        branch.yield () !17
                    } !18;
                    branch.yield ()
                } else 
                {
                  ^else_block7v1():
                    branch.yield ()
                };
                branch.yield ()
            };
            branch.return  !19
        }"#]].assert_eq(&after);
    Ok(())
}

/// Re-indexing must not change what the loop does, so it only fires when every
/// effect is already behind the divisibility guard.
#[test]
fn declines_when_an_effect_escapes_the_guard() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1).replace(
        "prod = math.i_mul",
        "memory.store (out, i) [] []: <(cube.ptr <cube.index , Global<0>>, cube.index ) -> ()>;
            prod = math.i_mul",
    );
    let (changed, _) = rewrite(&input)?;
    assert_eq!(changed, IRStatus::Unchanged);
    Ok(())
}

/// The new bounds are computed once, outside the loop, so the divisor and the
/// numerator have to be the same on every iteration.
#[test]
fn declines_when_the_divisor_varies_in_the_loop() -> Result<()> {
    let input = guarded_loop(9, 49, 48, 1, 8, 1).replace(
        "rem = math.u_rem (num, d)",
        "inner_d = math.i_add (d, i) [] []: <(cube.index , cube.index ) -> (cube.index )>;
            rem = math.u_rem (num, inner_d)",
    );
    let (changed, _) = rewrite(&input)?;
    assert_eq!(changed, IRStatus::Unchanged);
    Ok(())
}

/// A recovered `i` is only guaranteed to be an iteration the source loop had
/// when the source loop walked every integer in its range.
#[test]
fn declines_on_a_strided_loop() -> Result<()> {
    let (changed, _) = rewrite(&guarded_loop(9, 49, 48, 1, 8, 2))?;
    assert_eq!(changed, IRStatus::Unchanged);
    Ok(())
}

/// The rewritten loop must not look like a fresh match, or the pass would
/// chase its own output.
#[test]
fn is_idempotent() -> Result<()> {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let input = guarded_loop(9, 49, 48, 1, 8, 1);
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, &input).expect_ok(ctx);

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

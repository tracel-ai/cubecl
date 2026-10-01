use cubecl_ir::{
    NoSideEffects,
    dialect::memory::AddressSpaceAttr,
    interfaces::side_effects::{MemoryEffect, MemoryEffectsOp},
    prelude::*,
};
use cubecl_opt::analyses::dataflow_solver::{
    DataflowSolver, SolverConfig,
    dead_code::DeadCodeAnalysis,
    sccp::SparseConstantPropagationAnalysis,
    value_numbering::{ValueClassAnalysis, ValueNumberLattice, ValueNumberingAnalysis},
};
use expect_test::expect;
use itertools::Itertools;
use pliron::{
    common_traits::Named, init_env_logger_for_tests, irfmt::parsers::spaced,
    operation::verify_operation, parsable::parse_from_str, printable::Printable, result::ExpectOk,
};

#[pliron_op(
    name = "value_numbering.memory_def",
    format = "attr($def_address_space, $AddressSpaceAttr) `:` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (def_address_space: AddressSpaceAttr)
)]
#[op_traits(NoSideEffects)] // Has side effects, but should trigger memory check instead
pub struct MemoryDefInSpaceOp;

#[op_interface_impl]
impl MemoryEffectsOp for MemoryDefInSpaceOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_def_address_space(ctx).unwrap().0;
        vec![MemoryEffect::WriteAllInSpace(space)]
    }
}

#[pliron_op(
    name = "value_numbering.memory_use",
    format = "attr($use_address_space, $AddressSpaceAttr) `:` type($0)",
    interfaces = [NOpdsInterface<0>, OneResultInterface, NRegionsInterface<0>],
    verifier = "succ",
    attributes = (use_address_space: AddressSpaceAttr)
)]
#[op_traits(NoSideEffects)]
pub struct MemoryUseInSpaceOp;

#[op_interface_impl]
impl MemoryEffectsOp for MemoryUseInSpaceOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let space = self.get_attr_use_address_space(ctx).unwrap().0;
        vec![MemoryEffect::ReadAllInSpace(space)]
    }
}

fn run_numbering_on_text(ctx: &mut Context, input: &str) -> Result<DataflowSolver> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.load(ValueClassAnalysis::default());
    solver.load(ValueNumberingAnalysis::default());
    solver.initialize_and_run(ctx, op).unwrap();

    Ok(solver)
}

fn display_sorted_numbers(ctx: &Context, solver: &DataflowSolver) -> String {
    let mut states = solver.states_of_type::<ValueNumberLattice>();
    states.sort_by_key(|it| it.deref().anchor().id(ctx).to_string());
    states
        .iter()
        .map(|it| format!("{}\n", it.deref().disp(ctx)))
        .join("")
}

#[test]
fn value_numbering_basic_dependencies() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      abc = math.i_add (ab, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ab2 = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      abc2 = math.i_add (ab2, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return abc2
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `ab2` are equivalent.
    // - `abc` and `abc2` are equivalent because `ab` and `ab2` are equivalent.
    // - `ab` and `abc` are distinct.
    // - This verifies that an operation waits for the equivalence classes of
    //   its operands and that equivalent operand classes are propagated.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        c_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        abc_v4: ValueNumber(4)
        ab2_v5: ValueNumber(3)
        abc2_v6: ValueNumber(4)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_commutative_operands() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ba = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      ac = math.i_add (a, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ca = math.i_add (c, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      nested1 = math.i_add (ab, ac) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      nested2 = math.i_add (ca, ba) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return nested2
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `ba` are equivalent because `math.i_add` is commutative.
    // - `ac` and `ca` are equivalent for the same reason.
    // - `nested1` and `nested2` are equivalent because both pairs of operands
    //   resolve to the same equivalence classes after canonicalization.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        c_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        ba_v4: ValueNumber(3)
        ac_v5: ValueNumber(4)
        ca_v6: ValueNumber(4)
        nested1_v7: ValueNumber(5)
        nested2_v8: ValueNumber(5)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_non_commutative_operands() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.u_div (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ba = math.u_div (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ab2 = math.u_div (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return ab2
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `ab2` are equivalent.
    // - `ba` is distinct from `ab` because `math.u_div` is not commutative and
    //   operand order remains significant.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        ab_v2: ValueNumber(2)
        ba_v3: ValueNumber(3)
        ab2_v4: ValueNumber(2)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_inverse_comparisons() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32):
      lt_ab = cmp.u_less_than (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;
      gt_ba = cmp.u_greater_than (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;

      lt_ba = cmp.u_less_than (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;
      gt_ab = cmp.u_greater_than (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;

      branch.return lt_ab
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `lt_ab` and `gt_ba` are equivalent if comparison canonicalization
    //   normalizes an inverse predicate together with swapped operands.
    // - `lt_ba` and `gt_ab` are likewise equivalent.
    // - The two equivalence classes are distinct from each other.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        lt_ab_v2: ValueNumber(2)
        gt_ba_v3: ValueNumber(2)
        lt_ba_v4: ValueNumber(3)
        gt_ab_v5: ValueNumber(3)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_dependent_comparison_operands() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (cube.bool)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ba = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      cmp1 = cmp.u_less_than (ab, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;
      cmp2 = cmp.u_greater_than (c, ba) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;
      cmp3 = cmp.u_greater_than (ba, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (cube.bool)>;

      branch.return cmp1
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `ba` are equivalent because addition is commutative.
    // - `cmp1` and `cmp2` are equivalent if comparison canonicalization correctly
    //   normalizes the inverse predicate and swapped operands.
    // - `cmp3` is distinct from `cmp1`/`cmp2` because it has the opposite
    //   predicate/operand relationship.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        c_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        ba_v4: ValueNumber(3)
        cmp1_v5: ValueNumber(4)
        cmp2_v6: ValueNumber(4)
        cmp3_v7: ValueNumber(5)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_branch_convergence() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      cf.branch_conditional if cond ^left() else ^right()

      ^left():
      left_sum = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge()

      ^right():
      right_sum = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge()

      ^merge():
      merge_sum = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return merge_sum
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `left_sum`, `right_sum`, and `merge_sum` are all equivalent.
    // - This verifies that equivalent expressions discovered independently
    //   on different CFG paths retain the same class after convergence.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        left_sum_v3: ValueNumber(3)
        right_sum_v4: ValueNumber(3)
        merge_sum_v5: ValueNumber(3)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_branch_convergence_preserves_distinct_classes() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32, c: builtin.integer ui32):
      cf.branch_conditional if cond ^left() else ^right()

      ^left():
      left_ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      left_ac = math.i_add (a, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge()

      ^right():
      right_ab = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      right_ac = math.i_add (c, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge()

      ^merge():
      merge_ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      merge_ac = math.i_add (a, c) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return merge_ab
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `left_ab`, `right_ab`, and `merge_ab` are one equivalence class.
    // - `left_ac`, `right_ac`, and `merge_ac` are a different equivalence class.
    // - No value from the first class is equivalent to a value from the second.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        c_v3: ValueNumber(3)
        left_ab_v4: ValueNumber(4)
        left_ac_v5: ValueNumber(5)
        right_ab_v6: ValueNumber(4)
        right_ac_v7: ValueNumber(5)
        merge_ab_v8: ValueNumber(4)
        merge_ac_v9: ValueNumber(5)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_block_argument_propagates_equivalence() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      cf.branch_conditional if cond ^left(ab) else ^right(b)

      ^left(value_left: builtin.integer ui32):
      left_ab = math.i_add (value_left, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge(left_ab)

      ^right(value_right: builtin.integer ui32):
      right_ab = math.i_add (value_right, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge(right_ab)

      ^merge(value: builtin.integer ui32):
      result = math.i_add (value, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return result
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - The two incoming values for `merge.value` must have the same class only
    //   if the analysis can establish that they are equivalent.
    // - Here `left_ab` computes `(a + b) + b` while `right_ab` computes `b + a`;
    //   these are intentionally NOT equivalent.
    // - Therefore `merge.value` must receive a fresh/new class rather than
    //   propagating either predecessor's class.
    // - `result` must be distinct from either expression formed by substituting
    //   `left_ab` or `right_ab` for the block argument.
    //
    // This also exercises propagation through block arguments with non-identical
    // incoming equivalence classes.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        value_left_v4: ValueNumber(3)
        left_ab_v5: ValueNumber(4)
        value_right_v6: ValueNumber(2)
        right_ab_v7: ValueNumber(3)
        value_v8: ValueNumber(5)
        result_v9: ValueNumber(6)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_block_argument_propagates_matching_equivalence() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      cf.branch_conditional if cond ^left() else ^right()

      ^left():
      left_ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge(left_ab)

      ^right():
      right_ab = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge(right_ab)

      ^merge(value: builtin.integer ui32):
      ab_plus_b = math.i_add (ab, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      result = math.i_add (value, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return result
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `left_ab` and `right_ab` must be equivalent to `ab`, because i_add is
    //   commutative.
    // - The block argument `value` must receive that same equivalence class,
    //   since both incoming values have the same class.
    // - `result` must therefore be equivalent to the expression
    //   `math.i_add(ab, b)`.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        left_ab_v4: ValueNumber(3)
        right_ab_v5: ValueNumber(3)
        value_v6: ValueNumber(3)
        ab_plus_b_v7: ValueNumber(4)
        result_v8: ValueNumber(4)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_region_result_propagates_equivalence() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      result = scf.if cond : builtin.integer ui32 then {
        ^then():
        value = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (value)
      } else {
        ^else():
        other = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (other)
      };

      ab_plus_b = math.i_add (ab, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      after = math.i_add (result, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return after
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `value` and `other` must be equivalent to `ab`.
    // - The `scf.if` region result `result` must receive that same equivalence
    //   class because all region exits yield equivalent values.
    // - `after` must therefore be equivalent to `math.i_add(ab, b)`.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        value_v4: ValueNumber(3)
        other_v5: ValueNumber(3)
        result_v6: ValueNumber(3)
        ab_plus_b_v7: ValueNumber(4)
        after_v8: ValueNumber(4)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_region_result_mismatch_creates_new_class() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ac = math.i_add (a, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      result = scf.if cond : builtin.integer ui32 then {
        ^then():
        branch.yield (ab)
      } else {
        ^else():
        branch.yield (ac)
      };

      after = math.i_add (result, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return after
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `ac` must be distinct equivalence classes.
    // - The region result `result` must NOT receive either incoming class because
    //   the two region exits yield different equivalence classes.
    // - `result` must receive a fresh/new class representing the merged value.
    // - `after` must consequently be distinct from both `add(ab, b)` and
    //   `add(ac, b)`.
    expect![[r#"
        cond_v0: ValueNumber(0)
        a_v1: ValueNumber(1)
        b_v2: ValueNumber(2)
        ab_v3: ValueNumber(3)
        ac_v4: ValueNumber(4)
        result_v5: ValueNumber(5)
        after_v6: ValueNumber(6)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_ignores_folded_scf_if_branch() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      false = builtin.constant <cube.bool false> : cube.bool;

      result = scf.if false : builtin.integer ui32 then {
        ^then():
        divergent = math.u_div (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (divergent)
      } else {
        ^else():
        live = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (live)
      };

      branch.return result
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - `ab` and `live` are equivalent because `math.i_add` is commutative.
    // - The `then` region is unreachable because the `scf.if` condition is the
    //   constant `false`.
    // - The `divergent` value in the dead region must therefore not affect the
    //   equivalence of the `scf.if` result.
    // - `result` must receive the same equivalence class as `live` and `ab`.
    // - In particular, `result` must NOT receive a fresh class merely because
    //   the syntactically present dead region yields a different expression.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        ab_v2: ValueNumber(3)
        false_v3: ValueNumber(2)
        live_v5: ValueNumber(3)
        result_v6: ValueNumber(3)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_memory_no_alias() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32):
      b = value_numbering.memory_use Shared : builtin.integer ui32;
      c = value_numbering.memory_def Local : builtin.integer ui32;
      b2 = value_numbering.memory_use Shared : builtin.integer ui32;
      d = value_numbering.memory_def Local : builtin.integer ui32;
      b3 = value_numbering.memory_use Shared : builtin.integer ui32;

      branch.return a
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - Each memory def gets a fresh value number.
    // - `Local` definitions do not clobber `Shared` memory.
    // - `b`, `b2`, and `b3` therefore see the same MemorySSA value.
    // - Consequently `b`, `b2`, and `b3` have the same value number.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        c_v2: ValueNumber(2)
        b2_v3: ValueNumber(1)
        d_v4: ValueNumber(3)
        b3_v5: ValueNumber(1)
    "#]]
    .assert_eq(&display);
    Ok(())
}

#[test]
fn value_numbering_memory_clobber() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32):
      b = value_numbering.memory_use Shared : builtin.integer ui32;
      c = value_numbering.memory_def Shared : builtin.integer ui32;
      b2 = value_numbering.memory_use Shared : builtin.integer ui32;
      d = value_numbering.memory_def Shared : builtin.integer ui32;
      b3 = value_numbering.memory_use Shared : builtin.integer ui32;

      branch.return a
    }
  "#;

    let ctx = &mut Context::default();
    let solver = run_numbering_on_text(ctx, input)?;
    let display = display_sorted_numbers(ctx, &solver);

    // Expected:
    // - Each memory def gets a fresh value number.
    // - `c` creates a new Shared memory value, so `b2` is not equivalent to `b`.
    // - `d` creates another new Shared memory value, so `b3` is not equivalent
    //   to either `b` or `b2`.
    // - Thus all three Shared memory uses have distinct value numbers.
    expect![[r#"
        a_v0: ValueNumber(0)
        b_v1: ValueNumber(1)
        c_v2: ValueNumber(2)
        b2_v3: ValueNumber(3)
        d_v4: ValueNumber(4)
        b3_v5: ValueNumber(5)
    "#]]
    .assert_eq(&display);
    Ok(())
}

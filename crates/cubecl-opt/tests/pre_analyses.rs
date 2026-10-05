use cubecl_ir::{
    NoSideEffects,
    dialect::memory::AddressSpaceAttr,
    interfaces::side_effects::{MemoryEffect, MemoryEffectsOp},
    prelude::*,
};
use cubecl_opt::analyses::dataflow_solver::{
    AnalysisState, DataflowSolver, ProgramPoint, ReadRef,
    dense::{self, DenseLattice},
    pre::{AnticipatedExpressions, AvailableExpressions, Computes, compute_lcm_analyses},
};
use expect_test::expect;
use itertools::Itertools;
use pliron::{
    graph::HasLabel,
    init_env_logger_for_tests,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::{OpDbg, verify_operation},
    parsable::parse_from_str,
    printable::Printable,
    result::ExpectOk,
};

#[pliron_op(
    name = "pre.memory_def",
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
    name = "pre.memory_use",
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

fn run_pre_analyses_on_text(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let solver = compute_lcm_analyses(ctx, op)?;
    std::println!("{}", solver.disp(ctx));

    Ok(display_sets(ctx, &solver, op))
}

fn display_dense_fwd<T: dense::LatticeValue>(
    ctx: &Context,
    lattice: &ReadRef<DenseLattice<T>>,
) -> String {
    let lattice = lattice.deref();
    match lattice.anchor().prev_op(ctx) {
        Some(op) => format!("{}: {}\n", OpDbg { op, ctx }, lattice.value().disp(ctx)),
        None => format!(
            "block_start(^{}): {}\n",
            lattice.anchor().block().unwrap().label(ctx),
            lattice.value().disp(ctx)
        ),
    }
}

fn display_dense_bwd<T: dense::LatticeValue>(
    ctx: &Context,
    lattice: &ReadRef<DenseLattice<T>>,
) -> String {
    let lattice = lattice.deref();
    match lattice.anchor().next_op(ctx) {
        Some(op) => format!("{}: {}\n", OpDbg { op, ctx }, lattice.value().disp(ctx)),
        None => format!(
            "block_end(^{}): {}\n",
            lattice.anchor().block().unwrap().label(ctx),
            lattice.value().disp(ctx)
        ),
    }
}

// Reuse these for future dense analyses, but for now I'm putting them here
fn states_topological_fwd<'a, T: AnalysisState<Anchor = ProgramPoint>>(
    ctx: &Context,
    solver: &'a DataflowSolver,
    op: Ptr<Operation>,
    skip_outer_op: bool,
) -> Vec<ReadRef<'a, T>> {
    let mut out = Vec::new();
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            if let Some(state) = solver.lookup_state::<T>(ProgramPoint::at_block_start(ctx, block))
            {
                out.push(state);
            }
            for op in block.deref(ctx).iter(ctx) {
                out.extend(states_topological_fwd(ctx, solver, op, false));
            }
        }
    }
    if !skip_outer_op && let Some(state) = solver.lookup_state::<T>(ProgramPoint::after_op(ctx, op))
    {
        out.push(state);
    }
    out
}

fn states_topological_bwd<'a, T: AnalysisState<Anchor = ProgramPoint>>(
    ctx: &Context,
    solver: &'a DataflowSolver,
    op: Ptr<Operation>,
    skip_outer_op: bool,
) -> Vec<ReadRef<'a, T>> {
    let mut out = Vec::new();
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx).rev() {
            if let Some(state) = solver.lookup_state::<T>(ProgramPoint::at_block_end(ctx, block)) {
                out.push(state);
            }
            for op in block.deref(ctx).iter(ctx).rev() {
                out.extend(states_topological_bwd(ctx, solver, op, false));
            }
        }
    }
    if !skip_outer_op
        && let Some(state) = solver.lookup_state::<T>(ProgramPoint::before_op(ctx, op))
    {
        out.push(state);
    }
    out
}

fn display_sorted_computes(ctx: &Context, solver: &DataflowSolver, op: Ptr<Operation>) -> String {
    let states = states_topological_fwd::<Computes>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_fwd(ctx, it)).join("")
}

fn display_sorted_available(ctx: &Context, solver: &DataflowSolver, op: Ptr<Operation>) -> String {
    let states = states_topological_fwd::<AvailableExpressions>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_fwd(ctx, it)).join("")
}

fn display_sorted_anticipated(
    ctx: &Context,
    solver: &DataflowSolver,
    op: Ptr<Operation>,
) -> String {
    let states = states_topological_bwd::<AnticipatedExpressions>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_bwd(ctx, it)).join("")
}

fn display_sets(ctx: &Context, solver: &DataflowSolver, op: Ptr<Operation>) -> String {
    format!(
        "computes:
{}
available:
{}
anticipated:
{}",
        display_sorted_computes(ctx, solver, op),
        display_sorted_available(ctx, solver, op),
        display_sorted_anticipated(ctx, solver, op)
    )
}

#[test]
fn pre_basic() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      ab2 = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return ab2
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    expect![[r#"
        computes:
        ab_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        ab2_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        branch.return ab2_v3: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        ab_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2})
        ab2_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2})
        branch.return ab2_v3: Available({e2})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return ab2_v3: Anticipated({})
        ab2_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e2})
        ab_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e2})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_basic_dependencies() -> Result<()> {
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
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: `ab`/`ab2` must compute the same value number, and `abc`/`abc2`
    // must likewise compute the same value number; unrelated expressions stay distinct.
    // `Available`: after `ab`, its value must be available through `abc` and remain
    // available at `ab2`; after `abc`, both intermediate expression values must be available.
    // `Anticipated`: walking backward, `abc2` makes its expression anticipated before
    // `abc2`, and `ab2` makes both expressions anticipated; expressions
    // already computed in the forward direction must still be represented correctly.
    expect![[r#"
        computes:
        ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        ab2_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        abc2_v6 = math.i_add (ab2_v5, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return abc2_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3, e4})
        ab2_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3, e4})
        abc2_v6 = math.i_add (ab2_v5, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3, e4})
        branch.return abc2_v6: Available({e3, e4})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return abc2_v6: Anticipated({})
        abc2_v6 = math.i_add (ab2_v5, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4})
        ab2_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
        abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
        ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_non_commutative_expressions() -> Result<()> {
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
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: `ab` and `ab2` must have the same value number, while `ba` must
    // have a distinct number because operand order is significant for division.
    // `Available`: the value computed by `ab` must remain available through `ba`,
    // after `ba` both values should be available.
    // `Anticipated`: each division computation must make its own expression anticipated
    // immediately before it; the two operand orderings must remain distinct.
    expect![[r#"
        computes:
        ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        ab2_v4 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        branch.return ab2_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2})
        ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2, e3})
        ab2_v4 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2, e3})
        branch.return ab2_v4: Available({e2, e3})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return ab2_v4: Anticipated({})
        ab2_v4 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e2})
        ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e2, e3})
        ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e2, e3})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_branch_convergence() -> Result<()> {
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
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: all three additions must belong to the same value-number class
    // because the operand order is canonicalized as equivalent.
    // `Available`: the expression must be available at the merge because every incoming
    // branch computes the same value; it must not be incorrectly lost at the CFG join.
    // `Anticipated`: the expression must be anticipated before both branch computations
    // and remain anticipated through the merge computation.
    expect![[r#"
        computes:
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Computes({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        merge_sum_v5 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.return merge_sum_v5: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({})
        block_start(^left_block4v1): Available({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        cf.branch ^merge_block3v3(): Available({e3})
        block_start(^right_block5v1): Available({})
        right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        cf.branch ^merge_block3v3(): Available({e3})
        block_start(^merge_block3v3): Available({e3})
        merge_sum_v5 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        branch.return merge_sum_v5: Available({e3})

        anticipated:
        block_end(^merge_block3v3): Anticipated({})
        branch.return merge_sum_v5: Anticipated({})
        merge_sum_v5 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3})
        block_end(^right_block5v1): Anticipated({e3})
        cf.branch ^merge_block3v3(): Anticipated({e3})
        right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3})
        block_end(^left_block4v1): Anticipated({e3})
        cf.branch ^merge_block3v3(): Anticipated({e3})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3})
        block_end(^entry_block1v1): Anticipated({e3})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Anticipated({e3})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_partial_redundancy() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      cf.branch_conditional if cond ^left() else ^right()

      ^left():
      left_sum = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      cf.branch ^merge()

      ^right():
      cf.branch ^merge()

      ^merge():
      merge_sum = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return merge_sum
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: the expression must be computed in `left` and again in `merge`,
    // with both occurrences receiving the same value number.
    // `Available`: the expression must be available after the `left` computation but
    // not on the `right` path, so it must be absent at the merge's block entry.
    // `Anticipated`: the expression must be anticipated on both branches because the
    // merge computation is reached from either predecessor without an intervening kill.
    expect![[r#"
        computes:
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Computes({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        cf.branch ^merge_block3v3(): Computes({})
        merge_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.return merge_sum_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({})
        block_start(^left_block4v1): Available({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        cf.branch ^merge_block3v3(): Available({e3})
        block_start(^right_block5v1): Available({})
        cf.branch ^merge_block3v3(): Available({})
        block_start(^merge_block3v3): Available({})
        merge_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        branch.return merge_sum_v4: Available({e3})

        anticipated:
        block_end(^merge_block3v3): Anticipated({})
        branch.return merge_sum_v4: Anticipated({})
        merge_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3})
        block_end(^right_block5v1): Anticipated({e3})
        cf.branch ^merge_block3v3(): Anticipated({e3})
        block_end(^left_block4v1): Anticipated({e3})
        cf.branch ^merge_block3v3(): Anticipated({e3})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3})
        block_end(^entry_block1v1): Anticipated({e3})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Anticipated({e3})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_distinct_branch_expressions() -> Result<()> {
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
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: the `a + b` occurrences must form one class and the `a + c`
    // occurrences must form another; the two classes must remain distinct.
    // `Available`: both expression classes must be available at the merge because
    // every predecessor computes the corresponding expression.
    // `Anticipated`: both expressions must be anticipated before both branch computations
    // and remain so in the entry.
    expect![[r#"
        computes:
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Computes({})
        left_ab_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        left_ac_v5 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e5})
        cf.branch ^merge_block3v3(): Computes({})
        right_ab_v6 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        right_ac_v7 = math.i_add (c_v3, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e5})
        cf.branch ^merge_block3v3(): Computes({})
        merge_ab_v8 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        merge_ac_v9 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e5})
        branch.return merge_ab_v8: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({})
        block_start(^left_block4v1): Available({})
        left_ab_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4})
        left_ac_v5 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4, e5})
        cf.branch ^merge_block3v3(): Available({e4, e5})
        block_start(^right_block5v1): Available({})
        right_ab_v6 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4})
        right_ac_v7 = math.i_add (c_v3, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4, e5})
        cf.branch ^merge_block3v3(): Available({e4, e5})
        block_start(^merge_block3v3): Available({e4, e5})
        merge_ab_v8 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4, e5})
        merge_ac_v9 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4, e5})
        branch.return merge_ab_v8: Available({e4, e5})

        anticipated:
        block_end(^merge_block3v3): Anticipated({})
        branch.return merge_ab_v8: Anticipated({})
        merge_ac_v9 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e5})
        merge_ab_v8 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e5})
        block_end(^right_block5v1): Anticipated({e4, e5})
        cf.branch ^merge_block3v3(): Anticipated({e4, e5})
        right_ac_v7 = math.i_add (c_v3, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e5})
        right_ab_v6 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e5})
        block_end(^left_block4v1): Anticipated({e4, e5})
        cf.branch ^merge_block3v3(): Anticipated({e4, e5})
        left_ac_v5 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e5})
        left_ab_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e5})
        block_end(^entry_block1v1): Anticipated({e4, e5})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Anticipated({e4, e5})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_memory_no_alias() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32):
      b = pre.memory_use Shared : builtin.integer ui32;
      c = pre.memory_def Local : builtin.integer ui32;
      b2 = pre.memory_use Shared : builtin.integer ui32;
      d = pre.memory_def Local : builtin.integer ui32;
      b3 = pre.memory_use Shared : builtin.integer ui32;

      branch.return a
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: all three Shared-memory uses must have the same value number because
    // the intervening Local memory definitions do not modify Shared memory.
    // `Available`: the value produced by `b` must remain available after both Local
    // definitions and therefore be available at `b2` and `b3`.
    // `Anticipated`: the Shared-memory expression must propagate backward through both
    // Local definitions because neither definition kills that expression.
    expect![[r#"
        computes:
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        c_v2 = pre.memory_def Local:builtin.integer ui32: Computes({e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        d_v4 = pre.memory_def Local:builtin.integer ui32: Computes({e3})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        branch.return a_v0: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Available({e1})
        c_v2 = pre.memory_def Local:builtin.integer ui32: Available({e1, e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Available({e1, e2})
        d_v4 = pre.memory_def Local:builtin.integer ui32: Available({e1, e2, e3})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e1, e2, e3})
        branch.return a_v0: Available({e1, e2, e3})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return a_v0: Anticipated({})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e1})
        d_v4 = pre.memory_def Local:builtin.integer ui32: Anticipated({e1, e3})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e1, e3})
        c_v2 = pre.memory_def Local:builtin.integer ui32: Anticipated({e1, e2, e3})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e1, e2, e3})
    "#]]
    .assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_memory_clobber() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32):
      b = pre.memory_use Shared : builtin.integer ui32;
      c = pre.memory_def Shared : builtin.integer ui32;
      b2 = pre.memory_use Shared : builtin.integer ui32;
      d = pre.memory_def Shared : builtin.integer ui32;
      b3 = pre.memory_use Shared : builtin.integer ui32;

      branch.return a
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: each Shared-memory use must have a distinct value number because each
    // Shared memory definition creates a new memory version between the uses.
    // `Available`: all distinct values must be available at the end
    // `Anticipated`: the expression represented by `b` must be killed at `c`, so its
    // anticipated set must not incorrectly propagate backward across that Shared def.
    expect![[r#"
        computes:
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        c_v2 = pre.memory_def Shared:builtin.integer ui32: Computes({e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Computes({e3})
        d_v4 = pre.memory_def Shared:builtin.integer ui32: Computes({e4})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e5})
        branch.return a_v0: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Available({e1})
        c_v2 = pre.memory_def Shared:builtin.integer ui32: Available({e1, e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Available({e1, e2, e3})
        d_v4 = pre.memory_def Shared:builtin.integer ui32: Available({e1, e2, e3, e4})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e1, e2, e3, e4, e5})
        branch.return a_v0: Available({e1, e2, e3, e4, e5})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return a_v0: Anticipated({})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e5})
        d_v4 = pre.memory_def Shared:builtin.integer ui32: Anticipated({e4})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e3, e4})
        c_v2 = pre.memory_def Shared:builtin.integer ui32: Anticipated({e2, e4})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e1, e2, e4})
    "#]]
    .assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_memory_clobber_with_dependents() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(a: builtin.integer ui32):
      c = pre.memory_def Shared : builtin.integer ui32;
      b = pre.memory_use Shared : builtin.integer ui32;
      direct = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      indirect = math.i_add (direct, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return indirect
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: `b`, `direct`, and `indirect` must each be computed, with `direct`
    // directly depending on `b` and `indirect` indirectly depending on `b` through `direct`.
    // `Available`: the Shared definition at `c` must not prevent `b` from becoming available
    // after `b`; the value from `b` must then remain available for `direct` and `indirect`
    // because there is no intervening memory clobber.
    // `Anticipated`: `indirect` must propagate backward through `direct` to `b`, making all
    // three expressions anticipated after `c`; when the analysis reaches the Shared definition
    // at `c`, its kill set must remove `b` and both its direct and indirect dependents, so none
    // of those values remain anticipated immediately before `c`.
    expect![[r#"
        computes:
        c_v1 = pre.memory_def Shared:builtin.integer ui32: Computes({e1})
        b_v2 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        direct_v3 = math.i_add (b_v2, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        indirect_v4 = math.i_add (direct_v3, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return indirect_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        c_v1 = pre.memory_def Shared:builtin.integer ui32: Available({e1})
        b_v2 = pre.memory_use Shared:builtin.integer ui32: Available({e1, e2})
        direct_v3 = math.i_add (b_v2, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e1, e2, e3})
        indirect_v4 = math.i_add (direct_v3, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e1, e2, e3, e4})
        branch.return indirect_v4: Available({e1, e2, e3, e4})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return indirect_v4: Anticipated({})
        indirect_v4 = math.i_add (direct_v3, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4})
        direct_v3 = math.i_add (b_v2, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
        b_v2 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e2, e3, e4})
        c_v1 = pre.memory_def Shared:builtin.integer ui32: Anticipated({e1})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_region_result_convergence() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):
      ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      result = scf.if cond : builtin.integer ui32 then {
        ^then():
        then_sum = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (then_sum)
      } else {
        ^else():
        else_sum = math.i_add (b, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (else_sum)
      };

      after = math.i_add (result, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return after
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: `then_sum`, `else_sum`, and the `scf.if` region result must all have
    // the same value number because both region exits produce equivalent values.
    // `Available`: the common expression must be available across both region exits,
    // so the merged region result can carry that same value class.
    // `Anticipated`: the common expression must be anticipated on both region branches
    // and therefore survive the backward meet across the region control-flow join.
    expect![[r#"
        computes:
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        then_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (then_sum_v4): Computes({})
        else_sum_v5 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (else_sum_v5): Computes({})
        after_v7 = math.i_add (result_v6, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return after_v7: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        block_start(^then_block2v1): Available({e3})
        then_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        branch.yield (then_sum_v4): Available({e3})
        block_start(^else_block3v1): Available({e3})
        else_sum_v5 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        branch.yield (else_sum_v5): Available({e3})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({e3})
        after_v7 = math.i_add (result_v6, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3, e4})
        branch.return after_v7: Available({e3, e4})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return after_v7: Anticipated({})
        after_v7 = math.i_add (result_v6, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4})
        block_end(^then_block2v1): Anticipated({e4})
        branch.yield (then_sum_v4): Anticipated({e4})
        then_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
        block_end(^else_block3v1): Anticipated({e4})
        branch.yield (else_sum_v5): Anticipated({e4})
        else_sum_v5 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Anticipated({e3, e4})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e4})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_region_result_distinct_values() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32, b: builtin.integer ui32):

      result = scf.if cond : builtin.integer ui32 then {
        ^then():
        ab = math.i_add (a, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (ab)
      } else {
        ^else():
        aa = math.i_add (a, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
        branch.yield (aa)
      };

      after = math.i_add (result, b) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;

      branch.return after
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: `ab` and `aa` must remain distinct value-number classes, and the
    // `scf.if` result must receive a fresh class because its two region exits disagree.
    // `Available`: neither incoming class may be treated as available after the region
    // because availability requires the same class on every incoming region exit.
    // `Anticipated`: the two distinct classes must not be incorrectly merged by the
    // backward meet at the region boundary.
    expect![[r#"
        computes:
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (ab_v3): Computes({})
        aa_v4 = math.i_add (a_v1, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.yield (aa_v4): Computes({})
        after_v6 = math.i_add (result_v5, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e6})
        branch.return after_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        block_start(^then_block2v1): Available({})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e3})
        branch.yield (ab_v3): Available({e3})
        block_start(^else_block3v1): Available({})
        aa_v4 = math.i_add (a_v1, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e4})
        branch.yield (aa_v4): Available({e4})
        result_v5 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({})
        after_v6 = math.i_add (result_v5, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e6})
        branch.return after_v6: Available({e6})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return after_v6: Anticipated({})
        after_v6 = math.i_add (result_v5, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e6})
        block_end(^then_block2v1): Anticipated({e6})
        branch.yield (ab_v3): Anticipated({e6})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e6})
        block_end(^else_block3v1): Anticipated({e6})
        branch.yield (aa_v4): Anticipated({e6})
        aa_v4 = math.i_add (a_v1, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e4, e6})
        result_v5 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Anticipated({e6})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn pre_region_memory_clobber() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.bool, builtin.integer ui32) -> (builtin.integer ui32)> [] {
      ^entry(cond: cube.bool, a: builtin.integer ui32):
      before = pre.memory_use Shared : builtin.integer ui32;

      result = scf.if cond : builtin.integer ui32 then {
        ^then():
        def = pre.memory_def Shared : builtin.integer ui32;
        inside = pre.memory_use Shared : builtin.integer ui32;
        branch.yield (inside)
      } else {
        ^else():
        other = pre.memory_use Shared : builtin.integer ui32;
        branch.yield (other)
      };

      dependent = math.i_add (result, a) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>;
      after = pre.memory_use Shared : builtin.integer ui32;

      branch.return result
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Computed`: the Shared-memory uses before/inside/after the region must be distinguished
    // according to the MemorySSA state, with the `inside` use on the `then` path seeing the
    // new Shared memory version created by `def`.
    // `Available`: the value for the `before` Shared use must survive `def`.
    // `Anticipated`: the `Shared` expression depending on the pre-region memory state must be
    // killed at the `then` branch's memory definition and must not propagate backward through
    // that definition as though the memory were unchanged.
    // `after` should be killed by the memory phi at the `scf.if` and not propagate into the
    // branches, `dependent` should be killed when reaching `def` inside the `scf.if`.
    expect![[r#"
        computes:
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Computes({e4})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Computes({e5})
        branch.yield (inside_v4): Computes({})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        branch.yield (other_v5): Computes({})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e7})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Computes({e3})
        branch.return result_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({})
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Available({e2})
        block_start(^then_block2v1): Available({e2})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Available({e2, e4})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Available({e2, e4, e5})
        branch.yield (inside_v4): Available({e2, e4, e5})
        block_start(^else_block3v1): Available({e2})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e2})
        branch.yield (other_v5): Available({e2})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({e2})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e2, e7})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Available({e2, e3, e7})
        branch.return result_v6: Available({e2, e3, e7})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return result_v6: Anticipated({})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e3})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e3, e7})
        block_end(^then_block2v1): Anticipated({e7})
        branch.yield (inside_v4): Anticipated({e7})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e5, e7})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Anticipated({e4})
        block_end(^else_block3v1): Anticipated({e7})
        branch.yield (other_v5): Anticipated({e7})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e2, e7})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Anticipated({})
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e2})
    "#]].assert_eq(&sets);

    Ok(())
}

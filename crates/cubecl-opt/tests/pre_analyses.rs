use cubecl_ir::{
    NoSideEffects,
    dialect::memory::AddressSpaceAttr,
    interfaces::{
        Deterministic,
        side_effects::{MemoryEffect, MemoryEffectsOp},
    },
    prelude::*,
};
use cubecl_opt::{
    SparseBitSet,
    analyses::dataflow_solver::{
        AnalysisState, DataflowSolver, ProgramPoint, ReadRef,
        dense::{self, DenseLattice},
        pre::{
            Anticipated, AnticipatedMode, Available, Computes, InsertMode, compute_lcm_analyses,
            compute_pruned_insertion_points,
        },
    },
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
    utils::table::IMap,
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
#[op_interfaces(Deterministic)]
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

    let solver = compute_lcm_analyses(ctx, op, AnticipatedMode::Classical)?;
    std::println!("{}", solver.disp(ctx));

    Ok(display_sets(ctx, &solver, op))
}

fn run_speculative_pre_analyses_on_text(ctx: &mut Context, input: &str) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let solver = compute_lcm_analyses(ctx, op, AnticipatedMode::Speculative)?;
    std::println!("{}", solver.disp(ctx));

    Ok(display_sorted_anticipated(ctx, &solver, op))
}

fn run_insert_analyses_on_text(
    ctx: &mut Context,
    input: &str,
    antic_mode: AnticipatedMode,
    insert_mode: InsertMode,
) -> Result<String> {
    init_env_logger_for_tests!();
    let op = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);

    verify_operation(op, ctx)?;

    let mut solver = compute_lcm_analyses(ctx, op, antic_mode)?;
    let insertion_points = compute_pruned_insertion_points(ctx, op, &mut solver, insert_mode)?;
    std::println!("{}\n", solver.disp(ctx));

    std::println!("{}", display_sets(ctx, &solver, op));

    Ok(display_sorted_insertions(ctx, &insertion_points, op))
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

fn display_insertions(ctx: &Context, anchor: &ProgramPoint, value: &SparseBitSet) -> String {
    let value = value.iter().map(|it| format!("e{it}")).join(", ");
    match anchor.next_op(ctx) {
        Some(op) if let Some(block) = op.deref(ctx).get_parent_block() => format!(
            "^{}({}): {{{}}}\n",
            block.label(ctx),
            OpDbg { op, ctx },
            value
        ),
        Some(op) => format!("{}: {{{}}}\n", OpDbg { op, ctx }, value),
        None => format!(
            "block_end(^{}): {}\n",
            anchor.block().unwrap().label(ctx),
            value
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

fn insertions_topological<'a>(
    ctx: &Context,
    points: &'a IMap<ProgramPoint, SparseBitSet>,
    op: Ptr<Operation>,
) -> Vec<(ProgramPoint, &'a SparseBitSet)> {
    let mut out = Vec::new();
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            if let Some(state) = points.get(&ProgramPoint::at_block_end(ctx, block)) {
                out.push((ProgramPoint::at_block_end(ctx, block), state));
            }
            for op in block.deref(ctx).iter(ctx) {
                out.extend(insertions_topological(ctx, points, op));
            }
        }
    }
    if let Some(state) = points.get(&ProgramPoint::before_op(ctx, op)) {
        out.push((ProgramPoint::before_op(ctx, op), state));
    }
    out
}

fn display_sorted_computes(ctx: &Context, solver: &DataflowSolver, op: Ptr<Operation>) -> String {
    let states = states_topological_fwd::<Computes>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_fwd(ctx, it)).join("")
}

fn display_sorted_available(ctx: &Context, solver: &DataflowSolver, op: Ptr<Operation>) -> String {
    let states = states_topological_fwd::<Available>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_fwd(ctx, it)).join("")
}

fn display_sorted_anticipated(
    ctx: &Context,
    solver: &DataflowSolver,
    op: Ptr<Operation>,
) -> String {
    let states = states_topological_bwd::<Anticipated>(ctx, solver, op, true);
    states.iter().map(|it| display_dense_bwd(ctx, it)).join("")
}

fn display_sorted_insertions(
    ctx: &Context,
    points: &IMap<ProgramPoint, SparseBitSet>,
    op: Ptr<Operation>,
) -> String {
    let states = insertions_topological(ctx, points, op);
    states
        .iter()
        .map(|(anchor, value)| display_insertions(ctx, anchor, value))
        .join("")
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
        block_start(^entry_block1v1): Computes({})
        ab_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        ab2_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        branch.return ab2_v3: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1})
        ab_v2 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2})
        ab2_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2})
        branch.return ab2_v3: Available({e0, e1, e2})

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
        block_start(^entry_block1v1): Computes({})
        ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        ab2_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        abc2_v6 = math.i_add (ab2_v5, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return abc2_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2})
        ab_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        abc_v4 = math.i_add (ab_v3, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        ab2_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        abc2_v6 = math.i_add (ab2_v5, c_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        branch.return abc2_v6: Available({e0, e1, e2, e3, e4})

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
        block_start(^entry_block1v1): Computes({})
        ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        ab2_v4 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e2})
        branch.return ab2_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1})
        ab_v2 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2})
        ba_v3 = math.u_div (b_v1, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        ab2_v4 = math.u_div (a_v0, b_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.return ab2_v4: Available({e0, e1, e2, e3})

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
        block_start(^entry_block1v1): Computes({})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Computes({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        merge_sum_v5 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.return merge_sum_v5: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({e0, e1, e2})
        block_start(^left_block4v1): Available({e0, e1, e2})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2, e3})
        block_start(^right_block5v1): Available({e0, e1, e2})
        right_sum_v4 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2, e3})
        block_start(^merge_block3v3): Available({e0, e1, e2, e3})
        merge_sum_v5 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.return merge_sum_v5: Available({e0, e1, e2, e3})

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
        block_start(^entry_block1v1): Computes({})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Computes({})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        cf.branch ^merge_block3v3(): Computes({})
        cf.branch ^merge_block3v3(): Computes({})
        merge_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.return merge_sum_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({e0, e1, e2})
        block_start(^left_block4v1): Available({e0, e1, e2})
        left_sum_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2, e3})
        block_start(^right_block5v1): Available({e0, e1, e2})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2})
        block_start(^merge_block3v3): Available({e0, e1, e2})
        merge_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.return merge_sum_v4: Available({e0, e1, e2, e3})

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
        block_start(^entry_block1v1): Computes({})
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
        block_start(^entry_block1v1): Available({e0, e1, e2, e3})
        cf.branch_conditional if cond_v0 ^left_block4v1() else ^right_block5v1(): Available({e0, e1, e2, e3})
        block_start(^left_block4v1): Available({e0, e1, e2, e3})
        left_ab_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        left_ac_v5 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4, e5})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2, e3, e4, e5})
        block_start(^right_block5v1): Available({e0, e1, e2, e3})
        right_ab_v6 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        right_ac_v7 = math.i_add (c_v3, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4, e5})
        cf.branch ^merge_block3v3(): Available({e0, e1, e2, e3, e4, e5})
        block_start(^merge_block3v3): Available({e0, e1, e2, e3, e4, e5})
        merge_ab_v8 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4, e5})
        merge_ac_v9 = math.i_add (a_v1, c_v3) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4, e5})
        branch.return merge_ab_v8: Available({e0, e1, e2, e3, e4, e5})

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
        block_start(^entry_block1v1): Computes({})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        c_v2 = pre.memory_def Local:builtin.integer ui32: Computes({e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        d_v4 = pre.memory_def Local:builtin.integer ui32: Computes({e3})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        branch.return a_v0: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1})
        c_v2 = pre.memory_def Local:builtin.integer ui32: Available({e0, e1, e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2})
        d_v4 = pre.memory_def Local:builtin.integer ui32: Available({e0, e1, e2, e3})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2, e3})
        branch.return a_v0: Available({e0, e1, e2, e3})

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
        block_start(^entry_block1v1): Computes({})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Computes({e1})
        c_v2 = pre.memory_def Shared:builtin.integer ui32: Computes({e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Computes({e3})
        d_v4 = pre.memory_def Shared:builtin.integer ui32: Computes({e4})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e5})
        branch.return a_v0: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0})
        b_v1 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1})
        c_v2 = pre.memory_def Shared:builtin.integer ui32: Available({e0, e1, e2})
        b2_v3 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2, e3})
        d_v4 = pre.memory_def Shared:builtin.integer ui32: Available({e0, e1, e2, e3, e4})
        b3_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2, e3, e4, e5})
        branch.return a_v0: Available({e0, e1, e2, e3, e4, e5})

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
        block_start(^entry_block1v1): Computes({})
        c_v1 = pre.memory_def Shared:builtin.integer ui32: Computes({e1})
        b_v2 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        direct_v3 = math.i_add (b_v2, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        indirect_v4 = math.i_add (direct_v3, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return indirect_v4: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0})
        c_v1 = pre.memory_def Shared:builtin.integer ui32: Available({e0, e1})
        b_v2 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2})
        direct_v3 = math.i_add (b_v2, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        indirect_v4 = math.i_add (direct_v3, a_v0) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        branch.return indirect_v4: Available({e0, e1, e2, e3, e4})

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
        block_start(^entry_block1v1): Computes({})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        then_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (then_sum_v4): Computes({})
        else_sum_v5 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (else_sum_v5): Computes({})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Computes({e3})
        after_v7 = math.i_add (result_v6, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.return after_v7: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        block_start(^then_block2v1): Available({e0, e1, e2, e3})
        then_sum_v4 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.yield (then_sum_v4): Available({e0, e1, e2, e3})
        block_start(^else_block3v1): Available({e0, e1, e2, e3})
        else_sum_v5 = math.i_add (b_v2, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.yield (else_sum_v5): Available({e0, e1, e2, e3})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({e0, e1, e2, e3})
        after_v7 = math.i_add (result_v6, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3, e4})
        branch.return after_v7: Available({e0, e1, e2, e3, e4})

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
fn pre_loop_propagates_from_immediate_dominator() -> Result<()> {
    let input = r#"
    builtin.func @f: builtin.function <(cube.index, cube.index, cube.index, builtin.integer i64, builtin.integer i64) -> ()>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
    ^entry(start: cube.index, end: cube.index, step: cube.index, a: builtin.integer i64, b: builtin.integer i64):
        value_before = math.i_add (a, b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        scf.for start to end step step iter_args() {
        ^body(i: builtin.integer i32):
            branch.yield ()
        };
        value_after = math.i_sub (a, b) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
        branch.return
    }
  "#;

    let ctx = &mut Context::default();
    let sets = run_pre_analyses_on_text(ctx, input)?;

    // `Available` for `value_before` and `Anticipated` for `value_after` should propagate through
    // the loop.
    expect![[r#"
        computes:
        block_start(^entry_block1v1): Computes({})
        value_before_v5 = math.i_add (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Computes({e6})
        branch.yield (): Computes({})
        scf.for start_v0 to end_v1 step step_v2 {..}: Computes({})
        value_after_v7 = math.i_sub (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Computes({e7})
        branch.return : Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2, e3, e4})
        value_before_v5 = math.i_add (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Available({e0, e1, e2, e3, e4, e6})
        block_start(^body_block2v1): Available({e0, e1, e2, e3, e4, e5, e6})
        branch.yield (): Available({e0, e1, e2, e3, e4, e5, e6})
        scf.for start_v0 to end_v1 step step_v2 {..}: Available({e0, e1, e2, e3, e4, e6})
        value_after_v7 = math.i_sub (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Available({e0, e1, e2, e3, e4, e6, e7})
        branch.return : Available({e0, e1, e2, e3, e4, e6, e7})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return : Anticipated({})
        value_after_v7 = math.i_sub (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e7})
        block_end(^body_block2v1): Anticipated({e7})
        branch.yield (): Anticipated({e7})
        scf.for start_v0 to end_v1 step step_v2 {..}: Anticipated({e7})
        value_before_v5 = math.i_add (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e6, e7})
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
        block_start(^entry_block1v1): Computes({})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e3})
        branch.yield (ab_v3): Computes({})
        aa_v4 = math.i_add (a_v1, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e4})
        branch.yield (aa_v4): Computes({})
        result_v5 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Computes({e5})
        after_v6 = math.i_add (result_v5, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e6})
        branch.return after_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1, e2})
        block_start(^then_block2v1): Available({e0, e1, e2})
        ab_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e3})
        branch.yield (ab_v3): Available({e0, e1, e2, e3})
        block_start(^else_block3v1): Available({e0, e1, e2})
        aa_v4 = math.i_add (a_v1, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e4})
        branch.yield (aa_v4): Available({e0, e1, e2, e4})
        result_v5 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({e0, e1, e2, e5})
        after_v6 = math.i_add (result_v5, b_v2) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e5, e6})
        branch.return after_v6: Available({e0, e1, e2, e5, e6})

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
        block_start(^entry_block1v1): Computes({})
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Computes({e3})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Computes({e4})
        branch.yield (inside_v4): Computes({})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Computes({e2})
        branch.yield (other_v5): Computes({})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Computes({e6})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Computes({e7})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Computes({e5})
        branch.return result_v6: Computes({})

        available:
        block_start(^entry_block1v1): Available({e0, e1})
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2})
        block_start(^then_block2v1): Available({e0, e1, e2})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Available({e0, e1, e2, e3})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2, e3, e4})
        branch.yield (inside_v4): Available({e0, e1, e2, e3, e4})
        block_start(^else_block3v1): Available({e0, e1, e2})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2})
        branch.yield (other_v5): Available({e0, e1, e2})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Available({e0, e1, e2, e6})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Available({e0, e1, e2, e6, e7})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Available({e0, e1, e2, e5, e6, e7})
        branch.return result_v6: Available({e0, e1, e2, e5, e6, e7})

        anticipated:
        block_end(^entry_block1v1): Anticipated({})
        branch.return result_v6: Anticipated({})
        after_v8 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e5})
        dependent_v7 = math.i_add (result_v6, a_v1) [] []: <(builtin.integer ui32, builtin.integer ui32) -> (builtin.integer ui32)>: Anticipated({e5, e7})
        block_end(^then_block2v1): Anticipated({e7})
        branch.yield (inside_v4): Anticipated({e7})
        inside_v4 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e4, e7})
        def_v3 = pre.memory_def Shared:builtin.integer ui32: Anticipated({e3})
        block_end(^else_block3v1): Anticipated({e7})
        branch.yield (other_v5): Anticipated({e7})
        other_v5 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e2, e7})
        result_v6 = scf.if cond_v0 : builtin.integer ui32 then {..} else {..}: Anticipated({})
        before_v2 = pre.memory_use Shared:builtin.integer ui32: Anticipated({e2})
    "#]].assert_eq(&sets);

    Ok(())
}

#[test]
fn scm_if_uniform_condition() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(cube.bool, builtin.integer i64, builtin.integer i64) -> ()>
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(cond: cube.bool, a: builtin.integer i64, b: builtin.integer i64):
          result = scf.if cond : builtin.integer i64 then {
            ^then():
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield (value)
          } else {
            ^else():
              branch.yield (a)
          };
          branch.return
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should not* propagate through `if`
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return : Anticipated({})
        block_end(^then_block3v1): Anticipated({})
        branch.yield (value_v3): Anticipated({})
        value_v3 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e3})
        block_end(^else_block4v1): Anticipated({})
        branch.yield (a_v1): Anticipated({})
        result_v4 = scf.if cond_v0 : builtin.integer i64 then {..} else {..}: Anticipated({})
        builtin.func @f: builtin.function <(cube.bool , builtin.integer i64, builtin.integer i64) -> ()> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]{..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_if_divergent_condition() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64) -> ()> [] {
        ^entry(a: builtin.integer i64, b: builtin.integer i64):
          cond = plane.elect () [] []: <() -> (cube.bool)>;
          result = scf.if cond : builtin.integer i64 then {
            ^then():
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield (value)
          } else {
            ^else():
              branch.yield (a)
          };
          branch.return
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should* propagate through `if`
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return : Anticipated({})
        block_end(^then_block3v1): Anticipated({})
        branch.yield (value_v3): Anticipated({})
        value_v3 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e3})
        block_end(^else_block4v1): Anticipated({})
        branch.yield (a_v0): Anticipated({})
        result_v4 = scf.if cond_v2 : builtin.integer i64 then {..} else {..}: Anticipated({e3})
        cond_v2 = plane.elect () [] []: <() -> (cube.bool )>: Anticipated({e2, e3})
        builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64) -> ()> {..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_speculative_seed_external_fixpoint() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, cube.bool) -> ()>
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(a: builtin.integer i64, b: builtin.integer i64, uniform_cond: cube.bool):
          outer_cond = plane.elect () [] []: <() -> (cube.bool)>;
          result = scf.if outer_cond : builtin.integer i64 then {
            ^outer_then():
              middle = scf.if uniform_cond : builtin.integer i64 then {
                ^uniform_then():
                  inner_cond = plane.elect () [] []: <() -> (cube.bool)>;
                  inner = scf.if inner_cond : builtin.integer i64 then {
                    ^inner_then():
                      value = math.i_add (a, b) [] []:
                          <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
                      branch.yield (value)
                  } else {
                    ^inner_else():
                      branch.yield (a)
                  };
                  branch.yield (inner)
              } else {
                ^uniform_else():
                  fallback_value = math.i_add (a, b) [] []:
                      <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
                  branch.yield (fallback_value)
              };
              branch.yield (middle)
          } else {
            ^outer_else():
              branch.yield (a)
          };
          branch.return
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return : Anticipated({})
        block_end(^outer_then_block3v1): Anticipated({})
        branch.yield (middle_v8): Anticipated({})
        block_end(^uniform_then_block4v1): Anticipated({})
        branch.yield (inner_v6): Anticipated({})
        block_end(^inner_then_block5v1): Anticipated({})
        branch.yield (value_v5): Anticipated({})
        value_v5 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e5})
        block_end(^inner_else_block6v1): Anticipated({})
        branch.yield (a_v0): Anticipated({})
        inner_v6 = scf.if inner_cond_v4 : builtin.integer i64 then {..} else {..}: Anticipated({e5})
        inner_cond_v4 = plane.elect () [] []: <() -> (cube.bool )>: Anticipated({e4, e5})
        block_end(^uniform_else_block7v1): Anticipated({})
        branch.yield (fallback_value_v7): Anticipated({})
        fallback_value_v7 = math.i_add (a_v0, b_v1) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e5})
        middle_v8 = scf.if uniform_cond_v2 : builtin.integer i64 then {..} else {..}: Anticipated({e5})
        block_end(^outer_else_block8v1): Anticipated({})
        branch.yield (a_v0): Anticipated({})
        result_v9 = scf.if outer_cond_v3 : builtin.integer i64 then {..} else {..}: Anticipated({e5})
        outer_cond_v3 = plane.elect () [] []: <() -> (cube.bool )>: Anticipated({e3, e5})
        builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, cube.bool ) -> ()> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]{..}: Anticipated({})
    "#]].assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_for_uniform_bounds() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(cube.index, cube.index, cube.index, builtin.integer i64, builtin.integer i64) -> ()>
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(start: cube.index, end: cube.index, step: cube.index, a: builtin.integer i64, b: builtin.integer i64):
          scf.for start to end step step iter_args() {
            ^body(i: builtin.integer i32):
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield ()
          };
          branch.return
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should not* propagate before the loop
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return : Anticipated({})
        block_end(^body_block3v1): Anticipated({})
        branch.yield (): Anticipated({})
        value_v6 = math.i_add (a_v3, b_v4) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e6})
        scf.for start_v0 to end_v1 step step_v2 {..}: Anticipated({})
        builtin.func @f: builtin.function <(cube.index , cube.index , cube.index , builtin.integer i64, builtin.integer i64) -> ()> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]{..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_for_divergent_bounds() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(cube.index, cube.index, builtin.integer i64, builtin.integer i64) -> ()>
        [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(start: cube.index, step: cube.index, a: builtin.integer i64, b: builtin.integer i64):
          end = cube.read_builtin (UnitPos) : cube.index;
          scf.for start to end step step iter_args() {
            ^body(i: builtin.integer i32):
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield ()
          };
          branch.return
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should* propagate before the loop
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return : Anticipated({})
        block_end(^body_block3v1): Anticipated({})
        branch.yield (): Anticipated({})
        value_v6 = math.i_add (a_v2, b_v3) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e6})
        scf.for start_v0 to end_v4 step step_v1 {..}: Anticipated({e6})
        end_v4 = cube.read_builtin (UnitPos) : cube.index : Anticipated({e4, e6})
        builtin.func @f: builtin.function <(cube.index , cube.index , builtin.integer i64, builtin.integer i64) -> ()> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]{..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_while_uniform_condition() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(cube.bool, builtin.integer i64, builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>] {
        ^entry(cond: cube.bool, init: builtin.integer i64, a: builtin.integer i64, b: builtin.integer i64):
          y = scf.while init : builtin.integer i64 {
            ^before(x: builtin.integer i64):
              branch.condition (cond, x)
          } do {
            ^after(x2: builtin.integer i64):
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield (value)
          };
          branch.return y
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should not* propagate into ^before
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return y_v7: Anticipated({})
        block_end(^before_block3v1): Anticipated({})
        branch.condition (cond_v0, x_v4): Anticipated({})
        block_end(^after_block4v1): Anticipated({})
        branch.yield (value_v6): Anticipated({})
        value_v6 = math.i_add (a_v2, b_v3) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e4})
        y_v7 = scf.while init_v1 : builtin.integer i64 {..} do {..}: Anticipated({})
        builtin.func @f: builtin.function <(cube.bool , builtin.integer i64, builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> 
          [entry_point: cube.entrypoint_abi <cube_dim: (1, 1, 1)>]{..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn scm_while_divergent_condition() -> Result<()> {
    let input = r#"
    builtin.module @module {
      ^module_block():
      builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> [] {
        ^entry(init: builtin.integer i64, a: builtin.integer i64, b: builtin.integer i64):
          cond = plane.elect () [] []: <() -> (cube.bool)>;
          y = scf.while init : builtin.integer i64 {
            ^before(x: builtin.integer i64):
              branch.condition (cond, x)
          } do {
            ^after(x2: builtin.integer i64):
              value = math.i_add (a, b) [] []:
                  <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>;
              branch.yield (value)
          };
          branch.return y
      }
    }
  "#;

    let ctx = &mut Context::new();
    let actual = run_speculative_pre_analyses_on_text(ctx, input)?;

    // Anticipated *should* propagate into ^before, and out of the loop entirely
    expect![[r#"
        block_end(^module_block_block1v1): Anticipated({})
        block_end(^entry_block2v1): Anticipated({})
        branch.return y_v7: Anticipated({})
        block_end(^before_block3v1): Anticipated({e4})
        branch.condition (cond_v3, x_v4): Anticipated({e4})
        block_end(^after_block4v1): Anticipated({e4})
        branch.yield (value_v6): Anticipated({e4})
        value_v6 = math.i_add (a_v1, b_v2) [] []: <(builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)>: Anticipated({e4})
        y_v7 = scf.while init_v0 : builtin.integer i64 {..} do {..}: Anticipated({e4})
        cond_v3 = plane.elect () [] []: <() -> (cube.bool )>: Anticipated({e3, e4})
        builtin.func @f: builtin.function <(builtin.integer i64, builtin.integer i64, builtin.integer i64) -> (builtin.integer i64)> {..}: Anticipated({})
    "#]]
    .assert_eq(&actual);

    Ok(())
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

    let ctx = &mut Context::new();
    let actual =
        run_insert_analyses_on_text(ctx, input, AnticipatedMode::Classical, InsertMode::Lazy)?;

    // Should insert in `inner_else`
    expect![[r#"
        block_end(^inner_else_block7v1): e7
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_scm() -> Result<()> {
    let input = sscm_paper_example();

    let ctx = &mut Context::new();
    let actual =
        run_insert_analyses_on_text(ctx, input, AnticipatedMode::Classical, InsertMode::Sparse)?;

    // Should insert in `outer_else` before the `scf.if`
    expect![[r#"
        ^outer_else_block5v1(inner_v10 = scf.if inner_cond_v8 : builtin.integer i64 then {..} else {..}): {e7}
    "#]]
    .assert_eq(&actual);

    Ok(())
}

#[test]
fn insert_paper_example_sscm() -> Result<()> {
    let input = sscm_paper_example();

    let ctx = &mut Context::new();
    let actual =
        run_insert_analyses_on_text(ctx, input, AnticipatedMode::Speculative, InsertMode::Sparse)?;

    // Should insert in `entry` before the `scf.if`
    expect![[r#"
        ^entry_block2v1(result_v12 = scf.if outer_cond_v2 : builtin.integer i64 then {..} else {..}): {e7}
    "#]]
    .assert_eq(&actual);

    Ok(())
}

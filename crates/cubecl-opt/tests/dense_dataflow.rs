use cubecl_ir::prelude::*;
use cubecl_opt::analyses::dataflow_solver::{
    ChangeResult, DataflowSolver, ProgramPoint, ReadRef, SolverConfig, WriteRef,
    dense::{DenseBackward, DenseBackwardDataflowAnalysis, DenseLattice, LatticeValue},
};
use pliron::{
    irfmt::parsers::spaced, linked_list::ContainsLinkedList, operation::verify_operation,
    parsable::parse_from_str, printable::Printable, result::ExpectOk,
};

#[derive(Default, PartialEq)]
struct ReachesAdd(bool);

impl Printable for ReachesAdd {
    fn fmt(
        &self,
        _: &Context,
        _: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LatticeValue for ReachesAdd {
    fn meet(&mut self, other: &Self) -> ChangeResult {
        if !self.0 && other.0 {
            self.0 = true;
            ChangeResult::Changed
        } else {
            ChangeResult::Unchanged
        }
    }
}

struct ReachesAddBackward;

impl DenseBackwardDataflowAnalysis for ReachesAddBackward {
    type LatticeValue = ReachesAdd;

    fn visit_operation(
        _: &DenseBackward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        after: &ReadRef<DenseLattice<ReachesAdd>>,
        before: &WriteRef<DenseLattice<ReachesAdd>>,
    ) -> Result<()> {
        let reaches_add = Operation::is_op::<cubecl_ir::dialect::math::IAddOp>(op, ctx)
            || (after != before && after.deref().value().0);
        solver.update_state(ctx, before, |state| state.meet(&ReachesAdd(reaches_add)));
        Ok(())
    }

    fn set_to_exit_state(
        _: &DenseBackward<Self>,
        _: &DataflowSolver,
        _: &Context,
        _: &WriteRef<DenseLattice<ReachesAdd>>,
    ) {
    }
}

#[test]
fn dense_backward_revisits_a_branch_when_its_region_becomes_executable() -> Result<()> {
    let input = r#"
      builtin.func @f: builtin.function <(cube.index, cube.index) -> (cube.index)> [] {
        ^entry(arg: cube.index, other: cube.index):
          cond = cmp.u_less_than (arg, other) [] []: <(cube.index, cube.index) -> (cube.bool)>;
          result = scf.if cond : cube.index then {
            ^then():
              sum = math.i_add (arg, arg) [] []: <(cube.index, cube.index) -> (cube.index)>;
              branch.yield (sum)
          } else {
            ^else():
              branch.yield (arg)
          };
          branch.return result
      }
    "#;
    let ctx = &mut Context::new();
    let root = parse_from_str(spaced(Operation::top_level_parser()), ctx, input).expect_ok(ctx);
    verify_operation(root, ctx)?;
    let entry = root.deref(ctx).get_region(0).deref(ctx).get_head().unwrap();
    let branch = entry.deref(ctx).iter(ctx).nth(1).unwrap();
    let before_branch = ProgramPoint::before_op(ctx, branch);

    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DenseBackward::new(ReachesAddBackward));
    solver.initialize_and_run(ctx, root)?;

    let lattice = solver
        .lookup_state::<DenseLattice<ReachesAdd>>(before_branch)
        .unwrap();
    assert!(lattice.deref().value().0);
    Ok(())
}

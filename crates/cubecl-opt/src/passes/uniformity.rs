use cubecl_ir::prelude::*;
use pliron::printable::Printable;

use crate::analyses::dataflow_solver::{
    DataflowSolver, SolverConfig,
    control_flow_uniformity::BlockUniformityAnalysis,
    dead_code::DeadCodeAnalysis,
    sccp::SparseConstantPropagationAnalysis,
    value_uniformity::{DynamicUniformityAnalysis, StrictUniformityAnalysis},
};

pub fn uniformity(root_op: Ptr<Operation>, ctx: &mut Context) -> Result<IRStatus> {
    std::println!("IR before analysis: {}", root_op.disp(ctx));
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.load(BlockUniformityAnalysis);
    solver.load(StrictUniformityAnalysis::default());
    solver.load(DynamicUniformityAnalysis::default());
    solver.initialize_and_run(ctx, root_op)?;
    std::println!("Analysis state: {}", solver.disp(ctx));
    todo!();
}

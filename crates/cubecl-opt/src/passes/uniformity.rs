use cubecl_ir::{interfaces::uniformity::Uniformity, prelude::*};
use derive_more::Deref;
use pliron::{dict_key, value::DefiningEntity};

use crate::analyses::dataflow_solver::{
    DataflowSolver, SolverConfig,
    control_flow_uniformity::BlockUniformityAnalysis,
    dead_code::DeadCodeAnalysis,
    sccp::SparseConstantPropagationAnalysis,
    value_uniformity::{
        DynamicUniformityAnalysis, DynamicUniformityLattice, StrictUniformityAnalysis,
    },
};

dict_key!(DYNAMICALLY_UNIFORM_ATTR, "dynamically_uniform");

#[pliron_attr(name = "cube.uniform", format, verifier = "succ")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UniformAttr {
    Device,
    Cube,
    Plane,
}

pub const DEVICE_UNIFORM: UniformAttr = UniformAttr::Device;

pub fn op_dyn_uniformity(ctx: &Context, op: Ptr<Operation>) -> Option<UniformAttr> {
    op.get_attr(ctx, &DYNAMICALLY_UNIFORM_ATTR).map(|it| *it)
}

pub fn uniformity_solver(root_op: Ptr<Operation>, ctx: &Context) -> Result<DataflowSolver> {
    let mut solver = DataflowSolver::new(SolverConfig::default());
    solver.load(DeadCodeAnalysis::default());
    solver.load(SparseConstantPropagationAnalysis::default());
    solver.load(BlockUniformityAnalysis);
    solver.load(StrictUniformityAnalysis::default());
    solver.load(DynamicUniformityAnalysis::default());
    solver.initialize_and_run(ctx, root_op)?;
    Ok(solver)
}

#[derive(Deref)]
pub struct UniformityAnalysis(DataflowSolver);

#[pass_name]
impl Analysis for UniformityAnalysis {
    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let solver = uniformity_solver(op, ctx)?;
        Ok(Self(solver))
    }
}

pub struct MarkDynamicallyUniformPass;

#[pass_name]
impl Pass for MarkDynamicallyUniformPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        res.set_preserved::<UniformityAnalysis>();

        let analysis = analyses.get_analysis::<UniformityAnalysis>(op, ctx)?;
        let state = &mut (analysis, &mut res);

        visit_all_values(ctx, state, op, |ctx, (analysis, res), val| {
            if let Some(lattice) = analysis.lookup_state::<DynamicUniformityLattice>(val) {
                let attr = match lattice.deref().value().0 {
                    Uniformity::Uninitialized | Uniformity::None => return,
                    Uniformity::Device => UniformAttr::Device,
                    Uniformity::Cube => UniformAttr::Cube,
                    Uniformity::Plane => UniformAttr::Plane,
                };
                match val.defining_entity() {
                    DefiningEntity::Op(ptr) => {
                        ptr.set_attr(ctx, &DYNAMICALLY_UNIFORM_ATTR, attr);
                        res.ir_changed |= IRStatus::Changed;
                    }
                    DefiningEntity::Block(_) => {}
                }
            }
        });
        Ok(res)
    }
}

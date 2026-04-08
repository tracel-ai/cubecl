//! # `CubeCL` Optimizer
//!
//! A library that parses `CubeCL` IR into a
//! [control flow graph](https://en.wikipedia.org/wiki/Control-flow_graph), transforms it to
//! [static single-assignment form](https://en.wikipedia.org/wiki/Static_single-assignment_form)
//! and runs various optimizations on it.
//! The order of operations is as follows:
//!
//! 1. Parse root scope recursively into a [control flow graph](https://en.wikipedia.org/wiki/Control-flow_graph)
//! 2. Run optimizations that must be done before SSA transformation
//! 3. Analyze variable liveness
//! 4. Transform the graph to [pruned SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form#Pruned_SSA) form
//! 5. Run post-SSA optimizations and analyses in a loop until no more improvements are found
//! 6. Speed
//!
//! The output is represented as a [`petgraph`] graph of [`BasicBlock`]s terminated by [`ControlFlow`].
//! This can then be compiled into actual executable code by walking the graph and generating all
//! phi nodes, instructions and branches.
//!
//! # Representing [`PhiInstruction`] in non-SSA languages
//!
//! Phi instructions can be simulated by generating a mutable variable for each phi, then assigning
//! `value` to it in each relevant `block`.
//!

#![allow(unknown_lints, unnecessary_transmutes)]

use std::{
    collections::{HashMap, VecDeque},
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use analyses::{AnalysisCache, dominance::DomFrontiers, liveness::Liveness, writes::Writes};
use cubecl_core::CubeDim;
use cubecl_ir::{
    self as core, Allocator, Branch, Id, Operation, Operator, Processor, Scope, Type, Variable,
    VariableKind,
};
use gvn::GvnPass;
use passes::{
    CompositeMerge, ConstEval, ConstOperandSimplify, CopyTransform, DisaggregateArray,
    EliminateConstBranches, EliminateDeadBlocks, EliminateDeadPhi, EliminateUnusedVariables,
    EmptyBranchToSelect, InlineAssignments, MergeBlocks, MergeSameExpressions, OptimizerPass,
    ReduceStrength, RemoveIndexScalar,
};
use petgraph::{Direction, prelude::StableDiGraph, visit::EdgeRef};

mod analyses;
mod block;
mod control_flow;
mod debug;
mod gvn;
mod instructions;
mod passes;
mod phi_frontiers;
mod transformers;
mod version;

pub use analyses::uniformity::Uniformity;
pub use block::*;
pub use control_flow::*;
pub use petgraph::graph::{EdgeIndex, NodeIndex};
pub use transformers::*;
pub use version::PhiInstruction;

use crate::analyses::liveness::Captures;
pub use crate::analyses::liveness::shared::{SharedLiveness, SharedMemory};

/// An atomic counter with a simplified interface.
#[derive(Clone, Debug, Default)]
pub struct AtomicCounter {
    inner: Rc<AtomicUsize>,
}

impl AtomicCounter {
    /// Creates a new counter with `val` as its initial value.
    pub fn new(val: usize) -> Self {
        Self {
            inner: Rc::new(AtomicUsize::new(val)),
        }
    }

    /// Increments the counter and returns the last count.
    pub fn inc(&self) -> usize {
        self.inner.fetch_add(1, Ordering::AcqRel)
    }

    /// Gets the value of the counter without incrementing it.
    pub fn get(&self) -> usize {
        self.inner.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone)]
pub struct ConstArray {
    pub id: Id,
    pub length: usize,
    pub item: Type,
    pub values: Vec<core::Variable>,
}

#[derive(Default, Debug, Clone)]
pub struct Function {
    /// Explicit parameters passed to the function, i.e. the inputs to a closure
    pub explicit_params: Vec<Variable>,
    /// Implicit parameters passed to the function, i.e. kernel args, closure captures
    pub implicit_params: Vec<Variable>,
    pub const_arrays: Vec<ConstArray>,
    pub variables: HashMap<Id, Type>,
    pub graph: StableDiGraph<BasicBlock, u32>,
    pub root: NodeIndex,
    /// The single return block
    pub ret: NodeIndex,
    /// The return value, if any
    pub return_value: Option<Variable>,

    /// Analyses with persistent state
    analysis_cache: Rc<AnalysisCache>,
    /// The current block while parsing
    current_block: Option<NodeIndex>,
    /// The current loop's break target
    loop_break: VecDeque<NodeIndex>,
}

impl Deref for Function {
    type Target = StableDiGraph<BasicBlock, u32>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Function {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

type VarId = (Id, u16);

/// An optimizer that applies various analyses and optimization passes to the IR.
#[derive(Debug, Clone, Default)]
pub struct Optimizer {
    /// The overall program state
    pub main: Function,
    pub global_state: GlobalState,
}

#[derive(Debug, Clone)]
pub struct GlobalState {
    /// Allocator for kernel
    pub allocator: Allocator,
    /// Root scope to allocate variables on
    pub root_scope: Scope,
    pub extra_functions: HashMap<Id, Function>,
    /// The `CubeDim` used for range analysis
    pub(crate) cube_dim: CubeDim,
    pub(crate) transformers: Vec<Rc<dyn IrTransformer>>,
    pub(crate) processors: Rc<Vec<Box<dyn Processor>>>,
}

// Needed for WGPU server
unsafe impl Send for Optimizer {}
unsafe impl Sync for Optimizer {}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            allocator: Default::default(),
            root_scope: Scope::root(false),
            extra_functions: Default::default(),
            cube_dim: CubeDim::new_1d(1),
            transformers: Default::default(),
            processors: Default::default(),
        }
    }
}

impl Optimizer {
    /// Create a new optimizer with the scope, `CubeDim` and execution mode passed into the compiler.
    /// Parses the scope and runs several optimization and analysis loops.
    pub fn new(
        expand: Scope,
        cube_dim: CubeDim,
        transformers: Vec<Rc<dyn IrTransformer>>,
        processors: Vec<Box<dyn Processor>>,
    ) -> Self {
        let extra_funcs = expand.state().functions.clone();
        let mut global_state = GlobalState {
            allocator: expand.state().allocator.clone(),
            root_scope: expand.clone(),
            cube_dim,
            transformers,
            processors: Rc::new(processors),
            extra_functions: Default::default(),
        };
        for (id, func) in extra_funcs.into_iter() {
            let mut function = Function {
                explicit_params: func.explicit_params,
                return_value: func.scope.return_value,
                ..Default::default()
            };
            function.run_opt(&global_state, func.scope);
            global_state.extra_functions.insert(id, function);
        }
        let mut root_func = Function::default();
        root_func.run_opt(&global_state, expand);

        Self {
            global_state,
            main: root_func,
        }
    }

    /// Create a new optimizer with the scope, `CubeDim` and execution mode passed into the compiler.
    /// Parses the scope and runs several optimization and analysis loops.
    pub fn shared_only(expand: Scope, cube_dim: CubeDim) -> Self {
        let extra_funcs = expand.state().functions.clone();
        let mut global_state = GlobalState {
            allocator: expand.state().allocator.clone(),
            root_scope: expand.clone(),
            cube_dim,
            transformers: Vec::new(),
            processors: Rc::new(Vec::new()),
            extra_functions: Default::default(),
        };
        for (id, func) in extra_funcs.into_iter() {
            let mut function = Function {
                explicit_params: func.explicit_params,
                ..Default::default()
            };
            function.run_opt(&global_state, func.scope);
            global_state.extra_functions.insert(id, function);
        }
        let mut root_func = Function::default();
        root_func.run_opt(&global_state, expand);

        let mut opt = Self {
            global_state,
            main: root_func,
        };
        opt.run_shared_only();

        opt
    }

    /// Run only the shared memory analysis
    fn run_shared_only(&mut self) {
        self.main
            .parse_graph(&self.global_state, self.global_state.root_scope.clone());
        self.main.split_critical_edges();
        self.main
            .transform_ssa_and_merge_composites(&self.global_state);
        self.main.split_free();
        self.main.analysis::<SharedLiveness>(&self.global_state);
    }

    /// The entry block of the program
    pub fn entry(&self) -> NodeIndex {
        self.main.root
    }
}

/// Gets the `id` and `depth` of the variable if it's a `Local` and not atomic, `None` otherwise.
pub fn local_variable_id(variable: &core::Variable) -> Option<Id> {
    match variable.kind {
        core::VariableKind::LocalMut { id } if !variable.ty.is_atomic() => Some(id),
        _ => None,
    }
}

impl Function {
    fn parse_graph(&mut self, state: &GlobalState, scope: Scope) {
        let entry = self.add_node(BasicBlock::default());
        self.root = entry;
        self.current_block = Some(entry);
        self.ret = self.add_node(BasicBlock::default());
        *self[self.ret].control_flow.borrow_mut() = ControlFlow::Return {
            value: self.return_value,
        };
        self.parse_scope(state, scope);
        if let Some(current_block) = self.current_block {
            let ret = self.ret;
            self.add_edge(current_block, ret, 0);
        }
        // Analyses shouldn't have run at this point, but just in case they have, invalidate
        // all analyses that depend on the graph
        self.invalidate_structure();
    }

    /// Recursively parse a scope into the graph
    pub fn parse_scope(&mut self, state: &GlobalState, mut scope: Scope) -> bool {
        let processed = scope.process(state.processors.iter().map(|it| &**it));

        for var in processed.variables {
            if let VariableKind::LocalMut { id } = var.kind {
                self.variables.insert(id, var.ty);
            }
        }

        for (var, values) in scope.const_arrays.clone() {
            let VariableKind::ConstantArray {
                id,
                length,
                unroll_factor,
            } = var.kind
            else {
                unreachable!()
            };
            self.const_arrays.push(ConstArray {
                id,
                length: length * unroll_factor,
                item: var.ty,
                values,
            });
        }

        let is_break = processed.instructions.contains(&Branch::Break.into());

        for mut instruction in processed.instructions {
            let mut removed = false;
            for transform in state.transformers.iter() {
                match transform.maybe_transform(&mut scope, &instruction) {
                    TransformAction::Ignore => {}
                    TransformAction::Replace(replacement) => {
                        self.current_block_mut()
                            .ops
                            .borrow_mut()
                            .extend(replacement);
                        removed = true;
                        break;
                    }
                    TransformAction::Remove => {
                        removed = true;
                        break;
                    }
                }
            }
            if removed {
                continue;
            }
            match &mut instruction.operation {
                Operation::Branch(branch) => match self.parse_control_flow(state, branch.clone()) {
                    ControlFlowAction::None => {}
                    ControlFlowAction::AbortBlock => {
                        break;
                    }
                },
                _ => {
                    self.current_block_mut().ops.borrow_mut().push(instruction);
                }
            }
        }

        is_break
    }

    /// Remove non-constant index vectors from SSA transformation because they currently must be
    /// mutated
    fn exempt_index_assign_locals(&mut self) {
        for node in self.node_ids() {
            let ops = self[node].ops.clone();
            for op in ops.borrow().values() {
                if let Operation::Operator(Operator::IndexAssign(_)) = &op.operation
                    && let VariableKind::LocalMut { id } = &op.out().kind
                {
                    self.variables.remove(id);
                }
            }
        }
    }

    /// Mutable reference to the current basic block
    pub(crate) fn current_block_mut(&mut self) -> &mut BasicBlock {
        let current_block = self.current_block.unwrap();
        &mut self[current_block]
    }

    /// List of predecessor IDs of the `block`
    pub fn predecessors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.edges_directed(block, Direction::Incoming)
            .map(|it| it.source())
            .filter(|it| !self.is_unreachable(*it))
            .collect()
    }

    /// List of successor IDs of the `block`
    pub fn successors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.edges_directed(block, Direction::Outgoing)
            .map(|it| it.target())
            .collect()
    }

    /// Reference to the [`BasicBlock`] with ID `block`
    #[track_caller]
    pub fn block(&self, block: NodeIndex) -> &BasicBlock {
        &self[block]
    }

    /// Reference to the [`BasicBlock`] with ID `block`
    #[track_caller]
    pub fn block_mut(&mut self, block: NodeIndex) -> &mut BasicBlock {
        &mut self[block]
    }

    pub fn is_unreachable(&self, block: NodeIndex) -> bool {
        let control_flow = self[block].control_flow.borrow();
        matches!(*control_flow, ControlFlow::Unreachable)
    }

    /// A set of node indices for all blocks in the program
    pub fn node_ids(&self) -> Vec<NodeIndex> {
        self.node_indices().collect()
    }

    fn transform_ssa_and_merge_composites(&mut self, state: &GlobalState) {
        self.exempt_index_assign_locals();
        self.ssa_transform(state);

        let mut done = false;
        while !done {
            let changes = AtomicCounter::new(0);
            CompositeMerge.apply_post_ssa(self, state, changes.clone());
            if changes.get() > 0 {
                self.exempt_index_assign_locals();
                self.ssa_transform(state);
            } else {
                done = true;
            }
        }
    }

    fn ssa_transform(&mut self, state: &GlobalState) {
        self.place_phi_nodes(state);
        self.version_program(state);
        self.variables.clear();
        self.invalidate_analysis::<Writes>();
        self.invalidate_analysis::<DomFrontiers>();
    }

    /// Run all optimizations
    fn run_opt(&mut self, state: &GlobalState, scope: Scope) {
        self.parse_graph(state, scope);
        self.split_critical_edges();
        self.transform_ssa_and_merge_composites(state);
        self.apply_post_ssa_passes(state);

        // Special expensive passes that should only run once.
        // Need more optimization rounds in between.

        let arrays_prop = AtomicCounter::new(0);
        log::debug!("Applying {}", DisaggregateArray.name());
        DisaggregateArray.apply_post_ssa(self, state, arrays_prop.clone());
        if arrays_prop.get() > 0 {
            self.invalidate_analysis::<Liveness>();
            self.ssa_transform(state);
            self.apply_post_ssa_passes(state);
        }

        let gvn_count = AtomicCounter::new(0);
        log::debug!("Applying {}", GvnPass.name());
        GvnPass.apply_post_ssa(self, state, gvn_count.clone());
        log::debug!("Applying {}", ReduceStrength.name());
        ReduceStrength.apply_post_ssa(self, state, gvn_count.clone());
        log::debug!("Applying {}", CopyTransform.name());
        CopyTransform.apply_post_ssa(self, state, gvn_count.clone());

        if gvn_count.get() > 0 {
            self.apply_post_ssa_passes(state);
        }

        self.split_free();
        self.analysis::<SharedLiveness>(state);

        log::debug!("Applying {}", MergeBlocks.name());
        MergeBlocks.apply_post_ssa(self, state, AtomicCounter::new(0));

        log::debug!("Collecting captures");
        let captures = self.analysis::<Captures>(state);
        self.implicit_params = captures
            .at_block(self.root)
            .iter()
            .copied()
            .filter(|param| !self.explicit_params.contains(param))
            .collect();
    }

    fn apply_post_ssa_passes(&mut self, state: &GlobalState) {
        // Passes that run regardless of execution mode
        let mut passes: Vec<Box<dyn OptimizerPass>> = vec![
            Box::new(InlineAssignments),
            Box::new(EliminateUnusedVariables),
            Box::new(ConstOperandSimplify),
            Box::new(MergeSameExpressions),
            Box::new(ConstEval),
            Box::new(RemoveIndexScalar),
            Box::new(EliminateConstBranches),
            Box::new(EmptyBranchToSelect),
            Box::new(EliminateDeadBlocks),
            Box::new(EliminateDeadPhi),
        ];

        log::debug!("Applying post-SSA passes");
        loop {
            let counter = AtomicCounter::default();
            for pass in &mut passes {
                log::debug!("Applying {}", pass.name());
                pass.apply_post_ssa(self, state, counter.clone());
            }

            if counter.get() == 0 {
                break;
            }
        }
    }

    pub(crate) fn ret(&mut self) -> NodeIndex {
        if self[self.ret].block_use.contains(&BlockUse::Merge) {
            let ret = self.ret;
            let new_ret = self.add_node(BasicBlock::default());
            self.add_edge(new_ret, ret, 0);
            self.ret = new_ret;
            self.invalidate_structure();
            new_ret
        } else {
            self.ret
        }
    }

    pub fn const_arrays(&self) -> Vec<ConstArray> {
        self.const_arrays.clone()
    }

    pub fn all_params(&self) -> impl Iterator<Item = Variable> {
        self.explicit_params
            .iter()
            .copied()
            .chain(self.implicit_params.iter().copied())
    }
}

/// A visitor that does nothing.
pub fn visit_noop(_opt: &mut Function, _var: &mut Variable) {}

#[cfg(test)]
mod test {
    use cubecl_core as cubecl;
    use cubecl_core::cube;
    use cubecl_core::prelude::*;
    use cubecl_ir::{ElemType, ManagedVariable, Type, UIntKind, Variable, VariableKind};

    use crate::Optimizer;

    #[allow(unused)]
    #[cube(launch)]
    fn pre_kernel(x: u32, cond: u32, out: &mut Array<u32>) {
        let mut y = 0;
        let mut z = 0;
        if cond == 0 {
            y = x + 4;
        }
        z = x + 4;
        out[0] = y;
        out[1] = z;
    }

    #[test_log::test]
    #[ignore = "no good way to assert opt is applied"]
    fn test_pre() {
        let mut ctx = Scope::root(false);
        let x = ManagedVariable::Plain(Variable::new(
            VariableKind::GlobalScalar(0),
            Type::scalar(ElemType::UInt(UIntKind::U32)),
        ));
        let cond = ManagedVariable::Plain(Variable::new(
            VariableKind::GlobalScalar(1),
            Type::scalar(ElemType::UInt(UIntKind::U32)),
        ));
        let arr = ManagedVariable::Plain(Variable::new(
            VariableKind::GlobalOutputArray(0),
            Type::scalar(ElemType::UInt(UIntKind::U32)),
        ));

        pre_kernel::expand(&mut ctx, x.into(), cond.into(), arr.into());
        let opt = Optimizer::new(ctx, CubeDim::new_1d(1), vec![], vec![]);
        println!("{opt}")
    }
}

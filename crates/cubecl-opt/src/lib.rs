//! # CubeCL Optimizer
//!
//! A library that parses CubeCL IR into a
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
use cubecl_common::{CubeDim, ExecutionMode};
use cubecl_ir::{
    self as core, Allocator, Branch, Id, Item, Operation, Operator, Scope, Variable, VariableKind,
};
use gvn::GvnPass;
use passes::{
    CompositeMerge, ConstEval, ConstOperandSimplify, CopyPropagateArray, CopyTransform,
    EliminateConstBranches, EliminateDeadBlocks, EliminateDeadPhi, EliminateUnusedVariables,
    EmptyBranchToSelect, InlineAssignments, MergeBlocks, MergeSameExpressions, OptimizerPass,
    ReduceStrength, RemoveIndexScalar,
};
use petgraph::{
    Direction,
    dot::{Config, Dot},
    prelude::StableDiGraph,
    visit::EdgeRef,
};

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
    pub length: u32,
    pub item: Item,
    pub values: Vec<core::Variable>,
}

#[derive(Default, Debug, Clone)]
struct Program {
    pub const_arrays: Vec<ConstArray>,
    pub variables: HashMap<Id, Item>,
    pub graph: StableDiGraph<BasicBlock, u32>,
    root: NodeIndex,
}

impl Deref for Program {
    type Target = StableDiGraph<BasicBlock, u32>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Program {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

type VarId = (Id, u16);

/// An optimizer that applies various analyses and optimization passes to the IR.
#[derive(Debug, Clone)]
pub struct Optimizer {
    /// The overall program state
    program: Program,
    /// Allocator for kernel
    pub allocator: Allocator,
    /// Analyses with persistent state
    analysis_cache: Rc<AnalysisCache>,
    /// The current block while parsing
    current_block: Option<NodeIndex>,
    /// The current loop's break target
    loop_break: VecDeque<NodeIndex>,
    /// The single return block
    pub ret: NodeIndex,
    /// Root scope to allocate variables on
    pub root_scope: Scope,
    /// The `CubeDim` used for range analysis
    pub(crate) cube_dim: CubeDim,
    /// The execution mode, `Unchecked` skips bounds check optimizations.
    #[allow(unused)]
    pub(crate) mode: ExecutionMode,
    pub(crate) transformers: Vec<Rc<dyn IrTransformer>>,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            program: Default::default(),
            allocator: Default::default(),
            current_block: Default::default(),
            loop_break: Default::default(),
            ret: Default::default(),
            root_scope: Scope::root(false),
            cube_dim: Default::default(),
            mode: Default::default(),
            analysis_cache: Default::default(),
            transformers: Default::default(),
        }
    }
}

impl Optimizer {
    /// Create a new optimizer with the scope, `CubeDim` and execution mode passed into the compiler.
    /// Parses the scope and runs several optimization and analysis loops.
    pub fn new(
        expand: Scope,
        cube_dim: CubeDim,
        mode: ExecutionMode,
        transformers: Vec<Rc<dyn IrTransformer>>,
    ) -> Self {
        let mut opt = Self {
            root_scope: expand.clone(),
            cube_dim,
            mode,
            allocator: expand.allocator.clone(),
            transformers,
            ..Default::default()
        };
        opt.run_opt();

        opt
    }

    /// Run all optimizations
    fn run_opt(&mut self) {
        self.parse_graph(self.root_scope.clone());
        self.split_critical_edges();
        self.apply_pre_ssa_passes();
        self.exempt_index_assign_locals();
        self.ssa_transform();
        self.apply_post_ssa_passes();

        // Special expensive passes that should only run once.
        // Need more optimization rounds in between.

        let arrays_prop = AtomicCounter::new(0);
        CopyPropagateArray.apply_post_ssa(self, arrays_prop.clone());
        if arrays_prop.get() > 0 {
            self.invalidate_analysis::<Liveness>();
            self.ssa_transform();
            self.apply_post_ssa_passes();
        }

        let gvn_count = AtomicCounter::new(0);
        GvnPass.apply_post_ssa(self, gvn_count.clone());
        ReduceStrength.apply_post_ssa(self, gvn_count.clone());
        CopyTransform.apply_post_ssa(self, gvn_count.clone());

        if gvn_count.get() > 0 {
            self.apply_post_ssa_passes();
        }

        MergeBlocks.apply_post_ssa(self, AtomicCounter::new(0));
    }

    /// The entry block of the program
    pub fn entry(&self) -> NodeIndex {
        self.program.root
    }

    fn parse_graph(&mut self, scope: Scope) {
        let entry = self.program.add_node(BasicBlock::default());
        self.program.root = entry;
        self.current_block = Some(entry);
        self.ret = self.program.add_node(BasicBlock::default());
        *self.program[self.ret].control_flow.borrow_mut() = ControlFlow::Return;
        self.parse_scope(scope);
        if let Some(current_block) = self.current_block {
            self.program.add_edge(current_block, self.ret, 0);
        }
        // Analyses shouldn't have run at this point, but just in case they have, invalidate
        // all analyses that depend on the graph
        self.invalidate_structure();
    }

    fn apply_pre_ssa_passes(&mut self) {
        // Currently only one pre-ssa pass, but might add more
        let mut passes = vec![CompositeMerge];
        loop {
            let counter = AtomicCounter::default();

            for pass in &mut passes {
                pass.apply_pre_ssa(self, counter.clone());
            }

            if counter.get() == 0 {
                break;
            }
        }
    }

    fn apply_post_ssa_passes(&mut self) {
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

        loop {
            let counter = AtomicCounter::default();
            for pass in &mut passes {
                pass.apply_post_ssa(self, counter.clone());
            }

            if counter.get() == 0 {
                break;
            }
        }
    }

    /// Remove non-constant index vectors from SSA transformation because they currently must be
    /// mutated
    fn exempt_index_assign_locals(&mut self) {
        for node in self.node_ids() {
            let ops = self.program[node].ops.clone();
            for op in ops.borrow().values() {
                if let Operation::Operator(Operator::IndexAssign(_)) = &op.operation {
                    if let VariableKind::LocalMut { id } = &op.out().kind {
                        self.program.variables.remove(id);
                    }
                }
            }
        }
    }

    /// A set of node indices for all blocks in the program
    pub fn node_ids(&self) -> Vec<NodeIndex> {
        self.program.node_indices().collect()
    }

    fn ssa_transform(&mut self) {
        self.place_phi_nodes();
        self.version_program();
        self.program.variables.clear();
        self.invalidate_analysis::<Writes>();
        self.invalidate_analysis::<DomFrontiers>();
    }

    /// Mutable reference to the current basic block
    pub(crate) fn current_block_mut(&mut self) -> &mut BasicBlock {
        &mut self.program[self.current_block.unwrap()]
    }

    /// List of predecessor IDs of the `block`
    pub fn predecessors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.program
            .edges_directed(block, Direction::Incoming)
            .map(|it| it.source())
            .collect()
    }

    /// List of successor IDs of the `block`
    pub fn successors(&self, block: NodeIndex) -> Vec<NodeIndex> {
        self.program
            .edges_directed(block, Direction::Outgoing)
            .map(|it| it.target())
            .collect()
    }

    /// Reference to the [`BasicBlock`] with ID `block`
    #[track_caller]
    pub fn block(&self, block: NodeIndex) -> &BasicBlock {
        &self.program[block]
    }

    /// Reference to the [`BasicBlock`] with ID `block`
    #[track_caller]
    pub fn block_mut(&mut self, block: NodeIndex) -> &mut BasicBlock {
        &mut self.program[block]
    }

    /// Recursively parse a scope into the graph
    pub fn parse_scope(&mut self, mut scope: Scope) -> bool {
        let processed = scope.process();

        for var in processed.variables {
            if let VariableKind::LocalMut { id } = var.kind {
                self.program.variables.insert(id, var.item);
            }
        }

        for (var, values) in scope.const_arrays.clone() {
            let VariableKind::ConstantArray { id, length } = var.kind else {
                unreachable!()
            };
            self.program.const_arrays.push(ConstArray {
                id,
                length,
                item: var.item,
                values,
            });
        }

        let is_break = processed.instructions.contains(&Branch::Break.into());

        for mut instruction in processed.instructions {
            let mut removed = false;
            for transform in self.transformers.iter() {
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
                Operation::Branch(branch) => self.parse_control_flow(branch.clone()),
                _ => {
                    self.current_block_mut().ops.borrow_mut().push(instruction);
                }
            }
        }

        is_break
    }

    /// Gets the `id` and `depth` of the variable if it's a `Local` and not atomic, `None` otherwise.
    pub fn local_variable_id(&mut self, variable: &core::Variable) -> Option<Id> {
        match variable.kind {
            core::VariableKind::LocalMut { id } if !variable.item.elem.is_atomic() => Some(id),
            _ => None,
        }
    }

    pub(crate) fn ret(&mut self) -> NodeIndex {
        if self.program[self.ret].block_use.contains(&BlockUse::Merge) {
            let new_ret = self.program.add_node(BasicBlock::default());
            self.program.add_edge(new_ret, self.ret, 0);
            self.ret = new_ret;
            self.invalidate_structure();
            new_ret
        } else {
            self.ret
        }
    }

    pub fn const_arrays(&self) -> Vec<ConstArray> {
        self.program.const_arrays.clone()
    }

    pub fn dot_viz(&self) -> Dot<'_, &StableDiGraph<BasicBlock, u32>> {
        Dot::with_config(&self.program, &[Config::EdgeNoLabel])
    }
}

/// A visitor that does nothing.
pub fn visit_noop(_opt: &mut Optimizer, _var: &mut Variable) {}

#[cfg(test)]
mod test {
    use cubecl_core as cubecl;
    use cubecl_core::cube;
    use cubecl_core::prelude::*;
    use cubecl_ir::{Elem, ExpandElement, Item, UIntKind, Variable, VariableKind};

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

    #[test]
    #[ignore = "no good way to assert opt is applied"]
    fn test_pre() {
        let mut ctx = Scope::root(false);
        let x = ExpandElement::Plain(Variable::new(
            VariableKind::GlobalScalar(0),
            Item::new(Elem::UInt(UIntKind::U32)),
        ));
        let cond = ExpandElement::Plain(Variable::new(
            VariableKind::GlobalScalar(1),
            Item::new(Elem::UInt(UIntKind::U32)),
        ));
        let arr = ExpandElement::Plain(Variable::new(
            VariableKind::GlobalOutputArray(0),
            Item::new(Elem::UInt(UIntKind::U32)),
        ));

        pre_kernel::expand(&mut ctx, x.into(), cond.into(), arr.into());
        let opt = Optimizer::new(ctx, CubeDim::default(), ExecutionMode::Checked, vec![]);
        println!("{opt}")
    }
}

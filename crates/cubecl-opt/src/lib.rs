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

#![no_std]
#![allow(unknown_lints, unnecessary_transmutes)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

use core::{
    cell::RefCell,
    ops::{Deref, DerefMut},
};

use alloc::{boxed::Box, collections::vec_deque::VecDeque, rc::Rc, vec, vec::Vec};
use analyses::{AnalysisCache, dominance::DomFrontiers, liveness::Liveness, writes::LocalStores};
use cubecl_core::{
    CubeDim,
    post_processing::{
        analysis_helper::GlobalAnalyses,
        constant_prop::{ConstEval, ConstOperandSimplify},
        disaggregate::DisaggregateVisitor,
        visitor::InstructionVisitor,
    },
};
use cubecl_ir::{
    self as ir, AddressSpace, Allocator, Branch, Id, Instruction, Operation, Processor, Scope,
    Type, Value,
};
use gvn::GvnPass;
use hashbrown::HashMap;
use passes::{
    EliminateConstBranches, EliminateDeadBlocks, EliminateDeadPhi, EliminateUnusedVariables,
    EmptyBranchToSelect, InlineCopies, MergeBlocks, MergeSameExpressions, OptimizerPass,
    ReduceStrength,
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

pub(crate) use cubecl_core::post_processing::util::AtomicCounter;

pub use analyses::uniformity::Uniformity;
pub use block::*;
pub use control_flow::*;
pub use petgraph::graph::{EdgeIndex, NodeIndex};
pub use transformers::*;
pub use version::PhiInstruction;

pub use crate::analyses::liveness::shared::SharedLiveness;
use crate::{
    analyses::{dominance::Dominators, liveness::Captures, pointer_source::PointerSource},
    passes::{CopyTransform, DisaggregateArray},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryBlock {
    pub address_space: AddressSpace,
    pub value_ty: Type,
    pub alignment: usize,
    /// The root pointer value returned from the allocation or passed into the kernel. All other
    /// pointers into the same value are derived from this.
    pub root_ptr: Value,
}

impl MemoryBlock {
    /// The byte size of this shared memory
    pub fn size(&self) -> usize {
        self.value_ty.size()
    }
}

#[derive(Default, Debug, Clone)]
pub struct Function {
    /// Explicit parameters passed to the function, i.e. the inputs to a closure
    pub explicit_params: Vec<Value>,
    /// Implicit parameters passed to the function, i.e. kernel args, closure captures
    pub implicit_params: Vec<Value>,
    pub memories: HashMap<Id, MemoryBlock>,
    pub graph: StableDiGraph<BasicBlock, u32>,
    pub root: NodeIndex,
    /// The single return block
    pub ret: NodeIndex,
    /// The return value, if any
    pub return_value: Option<Value>,

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

/// An optimizer that applies various analyses and optimization passes to the IR.
#[derive(Debug, Clone, Default)]
pub struct Optimizer {
    pub main: Function,
    /// The overall program state
    pub global_state: GlobalState,
}

#[derive(Debug, Clone)]
pub struct GlobalState {
    /// Allocator for kernel
    pub allocator: Allocator,
    /// Root scope to allocate variables on
    pub root_scope: Scope,
    pub buffer_visibility: RefCell<Vec<BufferVisibility>>,
    pub extra_functions: HashMap<Id, Function>,
    /// The `CubeDim` used for range analysis
    pub(crate) cube_dim: CubeDim,
    pub(crate) transformers: Vec<Rc<dyn IrTransformer>>,
    pub(crate) processors: Rc<Vec<Box<dyn Processor>>>,
    pub(crate) visitors: Rc<RefCell<Vec<Box<dyn InstructionVisitor>>>>,
}

#[derive(Debug, Clone, Default)]
pub struct BufferVisibility {
    /// Whether the buffer is ever read from
    pub readable: bool,
    /// Whether the buffer is ever written to
    pub writable: bool,
}

// Needed for WGPU server
unsafe impl Send for Optimizer {}
unsafe impl Sync for Optimizer {}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            allocator: Default::default(),
            root_scope: Scope::root(false),
            buffer_visibility: Default::default(),
            extra_functions: Default::default(),
            cube_dim: CubeDim::new_1d(1),
            transformers: Default::default(),
            processors: Default::default(),
            visitors: Default::default(),
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
        visitors: Vec<Box<dyn InstructionVisitor>>,
        processors: Vec<Box<dyn Processor>>,
    ) -> Self {
        let extra_funcs = expand.state().functions.clone();
        let mut global_state = GlobalState {
            allocator: expand.state().allocator.clone(),
            root_scope: expand.clone(),
            buffer_visibility: Default::default(),
            cube_dim,
            transformers,
            processors: Rc::new(processors),
            visitors: Rc::new(RefCell::new(visitors)),
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
        let disaggregate: Box<dyn InstructionVisitor> = Box::new(DisaggregateVisitor::default());
        let mut global_state = GlobalState {
            allocator: expand.state().allocator.clone(),
            root_scope: expand.clone(),
            buffer_visibility: Default::default(),
            cube_dim,
            transformers: Vec::new(),
            processors: Rc::new(vec![]),
            visitors: Rc::new(RefCell::new(vec![disaggregate])),
            extra_functions: Default::default(),
        };
        for (id, func) in extra_funcs.into_iter() {
            let mut function = Function {
                explicit_params: func.explicit_params,
                ..Default::default()
            };
            function.run_shared_only(&global_state, func.scope);
            global_state.extra_functions.insert(id, function);
        }
        let mut root_func = Function::default();
        root_func.run_shared_only(&global_state, expand);

        Self {
            global_state,
            main: root_func,
        }
    }

    /// The entry block of the program
    pub fn entry(&self) -> NodeIndex {
        self.main.root
    }
}

impl GlobalState {
    fn set_buffer_readable(&self, id: Id) {
        let mut buffer_vis = self.buffer_visibility.borrow_mut();
        let idx = id as usize;
        if idx >= buffer_vis.len() {
            buffer_vis.resize(idx + 1, Default::default());
        }
        buffer_vis[idx].readable = true;
    }

    fn set_buffer_writable(&self, id: Id) {
        let mut buffer_vis = self.buffer_visibility.borrow_mut();
        let idx = id as usize;
        if idx >= buffer_vis.len() {
            buffer_vis.resize(idx + 1, Default::default());
        }
        buffer_vis[idx].writable = true;
    }
}

pub fn global_buffer_id(value: &ir::Value) -> Option<Id> {
    match value.address_space() {
        AddressSpace::Global(id) => Some(id),
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
    pub fn parse_scope(&mut self, state: &GlobalState, scope: Scope) -> bool {
        let global_analyses = GlobalAnalyses::default();
        global_analyses.recalculate_pointer_source(&scope);
        global_analyses.recalculate_used_values(&scope);
        for visitor in state.visitors.borrow_mut().iter_mut() {
            visitor.visit_scope(&scope, &global_analyses, &AtomicCounter::new(0));
        }
        let processed = scope.process(state.processors.iter().map(|it| &**it));

        for global_arg in processed.global_state.borrow().global_args.iter() {
            let address_space = global_arg.address_space();
            // Skip tensor maps
            if matches!(address_space, AddressSpace::Local) {
                continue;
            }
            self.memories.insert(
                global_arg.id(),
                MemoryBlock {
                    address_space,
                    value_ty: global_arg.ty.unwrap_ptr(),
                    alignment: global_arg.ty.align(),
                    root_ptr: *global_arg,
                },
            );
        }

        let is_break = processed.instructions.contains(&Branch::Break.into());

        for mut instruction in processed.instructions {
            let mut removed = false;
            for transform in state.transformers.iter() {
                match transform.maybe_transform(&scope, &instruction) {
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
                Operation::DeclareVariable {
                    value_ty,
                    addr_space,
                    alignment,
                } => {
                    let out = instruction.out.unwrap();
                    self.memories.insert(
                        out.id(),
                        MemoryBlock {
                            address_space: *addr_space,
                            value_ty: *value_ty,
                            alignment: *alignment,
                            root_ptr: out,
                        },
                    );
                    self.current_block_mut().ops.borrow_mut().push(instruction);
                }
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

    /// Return the breadth-first list of nodes along the dominator tree.
    /// This is useful for generating the blocks in a human-readable-ish order that follows the
    /// Vulkan spec (dominators before dominated).
    pub fn breadth_first_dominators(&self) -> Vec<NodeIndex> {
        self.analysis_cache
            .try_get::<Dominators>()
            .expect("Dominators should be present")
            .breadth_first_nodes()
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
        self.ssa_transform(state);

        let mut done = false;
        while !done {
            let changes = AtomicCounter::new(0);
            if changes.get() > 0 {
                self.ssa_transform(state);
            } else {
                done = true;
            }
        }
    }

    fn ssa_transform(&mut self, state: &GlobalState) {
        InlineCopies.apply_pre_ssa(self, state, AtomicCounter::new(0));
        self.place_phi_nodes(state);
        self.version_program(state);
        self.invalidate_analysis::<LocalStores>();
        self.invalidate_analysis::<DomFrontiers>();
    }

    /// Run all optimizations
    fn run_opt(&mut self, state: &GlobalState, scope: Scope) {
        self.parse_graph(state, scope);
        self.split_critical_edges();

        self.transform_ssa_and_merge_composites(state);
        self.analysis::<PointerSource>(state);
        self.apply_post_ssa_passes(state);

        // Special expensive passes that should only run once.
        // Need more optimization rounds in between.

        // Disaggregate arrays, remove the resulting pointer copies, then re-run mem2reg on the new
        // variables
        let arrays_prop = AtomicCounter::new(0);
        log::debug!("Applying {}", DisaggregateArray.name());
        DisaggregateArray.apply_post_ssa(self, state, arrays_prop.clone());
        if arrays_prop.get() > 0 {
            InlineCopies.apply_post_ssa(self, state, AtomicCounter::new(0));
            self.invalidate_analysis::<Liveness>();
            self.transform_ssa_and_merge_composites(state);
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

        self.update_buffer_vis(state);
        self.analysis::<Dominators>(state);
    }

    /// Run only the shared memory analysis
    fn run_shared_only(&mut self, state: &GlobalState, scope: Scope) {
        self.parse_graph(state, scope);
        self.split_critical_edges();
        self.transform_ssa_and_merge_composites(state);
        self.split_free();
        self.analysis::<PointerSource>(state);
        self.analysis::<SharedLiveness>(state);
        self.update_buffer_vis(state);
    }

    fn update_buffer_vis(&mut self, state: &GlobalState) {
        self.visit_all(
            state,
            |_, val| {
                if let Some(id) = global_buffer_id(val) {
                    state.set_buffer_readable(id);
                }
            },
            |_, val| {
                if let Some(id) = global_buffer_id(val) {
                    state.set_buffer_writable(id);
                }
            },
        );
    }

    fn apply_post_ssa_passes(&mut self, state: &GlobalState) {
        // Passes that run regardless of execution mode
        let mut passes: Vec<Box<dyn OptimizerPass>> = vec![
            Box::new(InlineCopies),
            Box::new(EliminateUnusedVariables),
            Box::new(ConstOperandSimplify),
            Box::new(MergeSameExpressions),
            Box::new(ConstEval),
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

    pub fn all_params(&self) -> impl Iterator<Item = Value> {
        self.explicit_params
            .iter()
            .copied()
            .chain(self.implicit_params.iter().copied())
    }

    pub fn create_local_mut(&mut self, state: &GlobalState, value_ty: Type) -> Value {
        let ty = Type::Pointer(value_ty.intern(), AddressSpace::Local);
        let val = state.allocator.create_value(ty);
        let root = self.root;
        self[root].ops.borrow_mut().push(Instruction::new(
            Operation::DeclareVariable {
                value_ty,
                addr_space: AddressSpace::Local,
                alignment: value_ty.align(),
            },
            val,
        ));
        self.memories.insert(
            val.id(),
            MemoryBlock {
                address_space: AddressSpace::Local,
                value_ty,
                alignment: value_ty.align(),
                root_ptr: val,
            },
        );
        val
    }
}

/// A visitor that does nothing.
pub fn visit_noop(_opt: &mut Function, _var: &mut Value) {}

#[cfg(test)]
mod test {
    use alloc::vec;
    use cubecl_core as cubecl;
    use cubecl_core::prelude::*;
    use cubecl_ir::{ElemType, Type, UIntKind};

    use crate::Optimizer;

    #[allow(unused)]
    #[cube(launch)]
    fn pre_kernel(x: u32, cond: u32, out: &mut [u32]) {
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
        let ctx = Scope::root(false);
        let u32 = Type::scalar(ElemType::UInt(UIntKind::U32));
        let x = ctx.create_value(u32).into();
        let cond = ctx.create_value(u32).into();
        let mut arr = ctx.global(0, u32).into();

        pre_kernel::expand(&ctx, x, cond, &mut arr);
        let opt = Optimizer::new(ctx, CubeDim::new_1d(1), vec![], vec![], vec![]);
        std::println!("{opt}")
    }
}

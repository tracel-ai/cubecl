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

use cubecl_ir::{AddressSpace, interfaces::TypedExt};

pub mod analyses;
// mod gvn;
pub mod passes;
pub mod scoped_map;

pub use analyses::uniformity::Uniformity;
use pliron::{context::Context, r#type::TypeHandle, value::Value};

pub use crate::analyses::liveness::shared::SharedLiveness;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryResource {
    pub address_space: AddressSpace,
    pub value_ty: TypeHandle,
    pub alignment: usize,
    /// The root pointer value returned from the allocation or passed into the kernel. All other
    /// pointers into the same value are derived from this.
    pub root_ptr: Value,
}

impl MemoryResource {
    /// The byte size of this shared memory
    pub fn size(&self, ctx: &Context) -> usize {
        self.value_ty.size(ctx)
    }
}

// /// An optimizer that applies various analyses and optimization passes to the IR.
// #[derive(Debug, Clone, Default)]
// pub struct Optimizer {
//     pub main: Function,
//     /// The overall program state
//     pub global_state: GlobalState,
// }

// #[derive(Debug, Clone)]
// pub struct GlobalState {
//     /// Allocator for kernel
//     pub allocator: Allocator,
//     /// Root scope to allocate variables on
//     pub root_scope: Scope,
//     pub buffer_visibility: RefCell<Vec<BufferVisibility>>,
//     /// The `CubeDim` used for range analysis
//     pub(crate) cube_dim: CubeDim,
// }

#[derive(Debug, Clone, Default)]
pub struct BufferVisibility {
    /// Whether the buffer is ever read from
    pub readable: bool,
    /// Whether the buffer is ever written to
    pub writable: bool,
}

// Needed for WGPU server
// unsafe impl Send for Optimizer {}
// unsafe impl Sync for Optimizer {}

// impl Default for GlobalState {
//     fn default() -> Self {
//         Self {
//             allocator: Default::default(),
//             root_scope: Scope::root(false),
//             buffer_visibility: Default::default(),
//             cube_dim: CubeDim::new_1d(1),
//         }
//     }
// }

// impl Optimizer {
//     /// Create a new optimizer with the scope, `CubeDim` and execution mode passed into the compiler.
//     /// Parses the scope and runs several optimization and analysis loops.
//     pub fn new(
//         expand: Scope,
//         cube_dim: CubeDim,
//         transformers: Vec<Rc<dyn IrTransformer>>,
//         visitors: Vec<Box<dyn InstructionVisitor>>,
//         processors: Vec<Box<dyn Processor>>,
//     ) -> Self {
//         let extra_funcs = expand.state().functions.clone();
//         let mut global_state = GlobalState {
//             allocator: expand.state().allocator.clone(),
//             root_scope: expand.clone(),
//             buffer_visibility: Default::default(),
//             cube_dim,
//             transformers,
//             processors: Rc::new(processors),
//             visitors: Rc::new(RefCell::new(visitors)),
//             extra_functions: Default::default(),
//         };
//         for (id, func) in extra_funcs.into_iter() {
//             let mut function = Function {
//                 explicit_params: func.explicit_params,
//                 return_value: func.scope.return_value,
//                 ..Default::default()
//             };
//             function.run_opt(&global_state, func.scope);
//             global_state.extra_functions.insert(id, function);
//         }
//         let mut root_func = Function::default();
//         root_func.run_opt(&global_state, expand);

//         Self {
//             global_state,
//             main: root_func,
//         }
//     }

//     /// Create a new optimizer with the scope, `CubeDim` and execution mode passed into the compiler.
//     /// Parses the scope and runs several optimization and analysis loops.
//     pub fn shared_only(expand: Scope, cube_dim: CubeDim) -> Self {
//         let extra_funcs = expand.state().functions.clone();
//         let disaggregate: Box<dyn InstructionVisitor> = Box::new(DisaggregateVisitor::default());
//         let mut global_state = GlobalState {
//             allocator: expand.state().allocator.clone(),
//             root_scope: expand.clone(),
//             buffer_visibility: Default::default(),
//             cube_dim,
//             transformers: Vec::new(),
//             processors: Rc::new(vec![]),
//             visitors: Rc::new(RefCell::new(vec![disaggregate])),
//             extra_functions: Default::default(),
//         };
//         for (id, func) in extra_funcs.into_iter() {
//             let mut function = Function {
//                 explicit_params: func.explicit_params,
//                 ..Default::default()
//             };
//             function.run_shared_only(&global_state, func.scope);
//             global_state.extra_functions.insert(id, function);
//         }
//         let mut root_func = Function::default();
//         root_func.run_shared_only(&global_state, expand);

//         Self {
//             global_state,
//             main: root_func,
//         }
//     }

//     /// The entry block of the program
//     pub fn entry(&self) -> NodeIndex {
//         self.main.root
//     }
// }

// impl Function {
//     /// Mutable reference to the current basic block
//     pub(crate) fn current_block_mut(&mut self) -> &mut BasicBlock {
//         let current_block = self.current_block.unwrap();
//         &mut self[current_block]
//     }

//     /// List of predecessor IDs of the `block`
//     pub fn predecessors(&self, block: NodeIndex) -> Vec<NodeIndex> {
//         self.edges_directed(block, Direction::Incoming)
//             .map(|it| it.source())
//             .filter(|it| !self.is_unreachable(*it))
//             .collect()
//     }

//     /// List of successor IDs of the `block`
//     pub fn successors(&self, block: NodeIndex) -> Vec<NodeIndex> {
//         self.edges_directed(block, Direction::Outgoing)
//             .map(|it| it.target())
//             .collect()
//     }

//     /// Return the breadth-first list of nodes along the dominator tree.
//     /// This is useful for generating the blocks in a human-readable-ish order that follows the
//     /// Vulkan spec (dominators before dominated).
//     pub fn breadth_first_dominators(&self) -> Vec<NodeIndex> {
//         self.analysis_cache
//             .try_get::<Dominators>()
//             .expect("Dominators should be present")
//             .breadth_first_nodes()
//     }

//     /// Reference to the [`BasicBlock`] with ID `block`
//     #[track_caller]
//     pub fn block(&self, block: NodeIndex) -> &BasicBlock {
//         &self[block]
//     }

//     /// Reference to the [`BasicBlock`] with ID `block`
//     #[track_caller]
//     pub fn block_mut(&mut self, block: NodeIndex) -> &mut BasicBlock {
//         &mut self[block]
//     }

//     pub fn is_unreachable(&self, block: NodeIndex) -> bool {
//         let control_flow = self[block].control_flow.borrow();
//         matches!(*control_flow, ControlFlow::Unreachable)
//     }

//     /// A set of node indices for all blocks in the program
//     pub fn node_ids(&self) -> Vec<NodeIndex> {
//         self.node_indices().collect()
//     }

//     fn transform_ssa_and_merge_composites(&mut self, state: &GlobalState) {
//         self.ssa_transform(state);

//         let mut done = false;
//         while !done {
//             let changes = AtomicCounter::new(0);
//             if changes.get() > 0 {
//                 self.ssa_transform(state);
//             } else {
//                 done = true;
//             }
//         }
//     }

//     fn ssa_transform(&mut self, state: &GlobalState) {
//         InlineCopies.apply_pre_ssa(self, state, AtomicCounter::new(0));
//         self.place_phi_nodes(state);
//         self.version_program(state);
//         self.invalidate_analysis::<LocalStores>();
//         self.invalidate_analysis::<DomFrontiers>();
//     }

//     /// Run all optimizations
//     fn run_opt(&mut self, state: &GlobalState, scope: Scope) {
//         self.parse_graph(state, scope);
//         self.split_critical_edges();

//         self.transform_ssa_and_merge_composites(state);
//         self.analysis::<PointerSource>(state);
//         self.apply_post_ssa_passes(state);

//         // Special expensive passes that should only run once.
//         // Need more optimization rounds in between.

//         // Disaggregate arrays, remove the resulting pointer copies, then re-run mem2reg on the new
//         // variables
//         let arrays_prop = AtomicCounter::new(0);
//         log::debug!("Applying {}", DisaggregateArray.name());
//         DisaggregateArray.apply_post_ssa(self, state, arrays_prop.clone());
//         if arrays_prop.get() > 0 {
//             InlineCopies.apply_post_ssa(self, state, AtomicCounter::new(0));
//             self.invalidate_analysis::<Liveness>();
//             self.transform_ssa_and_merge_composites(state);
//             self.apply_post_ssa_passes(state);
//         }

//         let gvn_count = AtomicCounter::new(0);
//         log::debug!("Applying {}", GvnPass.name());
//         GvnPass.apply_post_ssa(self, state, gvn_count.clone());
//         log::debug!("Applying {}", ReduceStrength.name());
//         ReduceStrength.apply_post_ssa(self, state, gvn_count.clone());
//         log::debug!("Applying {}", CopyTransform.name());
//         CopyTransform.apply_post_ssa(self, state, gvn_count.clone());

//         if gvn_count.get() > 0 {
//             self.apply_post_ssa_passes(state);
//         }

//         self.split_free();
//         self.analysis::<SharedLiveness>(state);

//         log::debug!("Applying {}", MergeBlocks.name());
//         MergeBlocks.apply_post_ssa(self, state, AtomicCounter::new(0));

//         log::debug!("Collecting captures");
//         let captures = self.analysis::<Captures>(state);
//         self.implicit_params = captures
//             .at_block(self.root)
//             .iter()
//             .copied()
//             .filter(|param| !self.explicit_params.contains(param))
//             .collect();

//         self.update_buffer_vis(state);
//         self.analysis::<Dominators>(state);
//     }

//     /// Run only the shared memory analysis
//     fn run_shared_only(&mut self, state: &GlobalState, scope: Scope) {
//         self.parse_graph(state, scope);
//         self.split_critical_edges();
//         self.transform_ssa_and_merge_composites(state);
//         self.split_free();
//         self.analysis::<PointerSource>(state);
//         self.analysis::<SharedLiveness>(state);
//         self.update_buffer_vis(state);
//     }

//     fn update_buffer_vis(&mut self, state: &GlobalState) {
//         self.visit_all(
//             state,
//             |_, val| {
//                 if let Some(id) = global_buffer_id(val) {
//                     state.set_buffer_readable(id);
//                 }
//             },
//             |_, val| {
//                 if let Some(id) = global_buffer_id(val) {
//                     state.set_buffer_writable(id);
//                 }
//             },
//         );
//     }

//     fn apply_post_ssa_passes(&mut self, state: &GlobalState) {
//         // Passes that run regardless of execution mode
//         let mut passes: Vec<Box<dyn OptimizerPass>> = vec![
//             Box::new(InlineCopies),
//             Box::new(EliminateUnusedVariables),
//             Box::new(ConstOperandSimplify),
//             Box::new(MergeSameExpressions),
//             Box::new(ConstEval),
//             Box::new(EliminateConstBranches),
//             Box::new(EmptyBranchToSelect),
//             Box::new(EliminateDeadBlocks),
//             Box::new(EliminateDeadPhi),
//         ];

//         log::debug!("Applying post-SSA passes");
//         loop {
//             let counter = AtomicCounter::default();
//             for pass in &mut passes {
//                 log::debug!("Applying {}", pass.name());
//                 pass.apply_post_ssa(self, state, counter.clone());
//             }

//             if counter.get() == 0 {
//                 break;
//             }
//         }
//     }

//     pub(crate) fn ret(&mut self) -> NodeIndex {
//         if self[self.ret].block_use.contains(&BlockUse::Merge) {
//             let ret = self.ret;
//             let new_ret = self.add_node(BasicBlock::default());
//             self.add_edge(new_ret, ret, 0);
//             self.ret = new_ret;
//             self.invalidate_structure();
//             new_ret
//         } else {
//             self.ret
//         }
//     }

//     pub fn all_params(&self) -> impl Iterator<Item = ExpandValue> {
//         self.explicit_params
//             .iter()
//             .copied()
//             .chain(self.implicit_params.iter().copied())
//     }

//     pub fn create_local_mut(&mut self, state: &GlobalState, value_ty: Type) -> ExpandValue {
//         let ty = Type::Pointer(value_ty.intern(), AddressSpace::Local);
//         let val = state.allocator.create_value(ty);
//         let root = self.root;
//         self[root].ops.borrow_mut().push(Instruction::new(
//             Operation::DeclareVariable {
//                 value_ty,
//                 addr_space: AddressSpace::Local,
//                 alignment: value_ty.align(),
//             },
//             val,
//         ));
//         self.memories.insert(
//             val.id(),
//             MemoryResource {
//                 address_space: AddressSpace::Local,
//                 value_ty,
//                 alignment: value_ty.align(),
//                 root_ptr: val,
//             },
//         );
//         val
//     }
// }

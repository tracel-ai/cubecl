use alloc::collections::vec_deque::VecDeque;
use cubecl_ir::Id;
use hashbrown::{HashMap, HashSet};
use petgraph::graph::NodeIndex;

use crate::{Function, GlobalState, analyses::post_order::PostOrder, local_variable_id};

use super::Analysis;

pub struct Liveness {
    live_vars: HashMap<NodeIndex, HashSet<Id>>,
}

#[derive(Clone)]
struct BlockSets {
    generated: HashSet<Id>,
    kill: HashSet<Id>,
}

struct State {
    worklist: VecDeque<NodeIndex>,
    block_sets: HashMap<NodeIndex, BlockSets>,
}

impl Analysis for Liveness {
    fn init(func: &mut Function, state: &GlobalState) -> Self {
        let mut this = Self::empty(func);
        this.analyze_liveness(func, state);
        this
    }
}

impl Liveness {
    pub fn empty(func: &Function) -> Self {
        let live_vars = func
            .node_ids()
            .iter()
            .map(|it| (*it, HashSet::new()))
            .collect();
        Self { live_vars }
    }

    pub fn at_block(&self, block: NodeIndex) -> &HashSet<Id> {
        &self.live_vars[&block]
    }

    pub fn is_dead(&self, node: NodeIndex, var: Id) -> bool {
        !self.at_block(node).contains(&var)
    }

    /// Do a conservative block level liveness analysis
    pub fn analyze_liveness(&mut self, func: &mut Function, global_state: &GlobalState) {
        let mut state = State {
            worklist: VecDeque::from(func.analysis::<PostOrder>(global_state).forward()),
            block_sets: HashMap::new(),
        };
        while let Some(block) = state.worklist.pop_front() {
            self.analyze_block(func, global_state, block, &mut state);
        }
    }

    fn analyze_block(
        &mut self,
        func: &mut Function,
        global_state: &GlobalState,
        block: NodeIndex,
        state: &mut State,
    ) {
        let BlockSets { generated, kill } = block_sets(func, global_state, block, state);

        let mut live_vars = generated.clone();

        for successor in func.successors(block) {
            let successor = &self.live_vars[&successor];
            live_vars.extend(successor.difference(kill));
        }

        if live_vars != self.live_vars[&block] {
            state.worklist.extend(func.predecessors(block));
            self.live_vars.insert(block, live_vars);
        }
    }
}

fn block_sets<'a>(
    func: &mut Function,
    global_state: &GlobalState,
    block: NodeIndex,
    state: &'a mut State,
) -> &'a BlockSets {
    let block_sets = state.block_sets.entry(block);
    block_sets.or_insert_with(|| calculate_block_sets(func, global_state, block))
}

fn calculate_block_sets(func: &mut Function, state: &GlobalState, block: NodeIndex) -> BlockSets {
    let mut generated = HashSet::new();
    let mut kill = HashSet::new();

    let ops = func[block].ops.clone();

    let control_flow = func[block].control_flow.clone();
    func.visit_control_flow(&mut control_flow.borrow_mut(), |_, var| {
        if let Some(id) = local_variable_id(var) {
            generated.insert(id);
        }
    });
    let mut ops = ops.borrow().clone();
    for op in ops.values_mut().rev() {
        // Reads must be tracked after writes
        func.visit_out(&mut op.out, |_, var| {
            if let Some(id) = local_variable_id(var) {
                kill.insert(id);
                generated.remove(&id);
            }
        });
        func.visit_operation(state, &mut op.operation, |_, var| {
            if let Some(id) = local_variable_id(var) {
                generated.insert(id);
            }
        });
    }

    BlockSets { generated, kill }
}

/// Shared memory liveness analysis and allocation
pub mod shared {
    use alloc::vec::Vec;
    use cubecl_ir::{Marker, Operation, Type, Variable, VariableKind};

    use crate::Uniformity;

    use super::*;

    /// A shared memory instance, all the information contained in the `VariableKind`, but with
    /// a non-optional `align`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SharedMemory {
        pub id: Id,
        pub ty: Type,
        pub align: usize,
    }

    impl SharedMemory {
        /// The byte size of this shared memory
        pub fn size(&self) -> usize {
            self.ty.size()
        }
    }

    /// A specific allocation of shared memory at some `offset`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SmemAllocation {
        /// The shared memory being allocated
        pub smem: SharedMemory,
        /// The offset in the shared memory buffer
        pub offset: usize,
    }

    /// Shared liveness works the other way around from normal liveness, since shared memory lives
    /// forever by default. So any use (read or write) inserts it as live, while only `free` changes
    /// the state to dead.
    ///
    /// It also handles allocation of slices to each shared memory object, using the analyzed
    /// liveness. `allocations` contains a specific slice allocation for each shared memory, while
    /// ensuring no shared memories that exist at the same time can overlap.
    #[derive(Default, Clone)]
    pub struct SharedLiveness {
        live_vars: HashMap<NodeIndex, HashSet<Id>>,
        /// Map of all shared memories by their ID. Populated during the first pass with all
        /// accessed shared memories.
        pub shared_memories: HashMap<Id, SharedMemory>,
        /// Map of allocations for each shared memory by its ID. Populated after the analysis, and
        /// should contain all memories from `shared_memories`.
        pub allocations: HashMap<Id, SmemAllocation>,
    }

    impl Analysis for SharedLiveness {
        fn init(func: &mut Function, state: &GlobalState) -> Self {
            let mut this = Self::empty(func);
            this.analyze_liveness(func, state);
            this.uniformize_liveness(func, state);
            this.allocate_slices(func);
            this
        }
    }

    impl SharedLiveness {
        pub fn empty(func: &Function) -> Self {
            let live_vars = func
                .node_ids()
                .iter()
                .map(|it| (*it, HashSet::new()))
                .collect();
            Self {
                live_vars,
                shared_memories: Default::default(),
                allocations: Default::default(),
            }
        }

        pub fn at_block(&self, block: NodeIndex) -> &HashSet<Id> {
            &self.live_vars[&block]
        }

        fn is_live(&self, node: NodeIndex, var: Id) -> bool {
            self.at_block(node).contains(&var)
        }

        /// Do a conservative block level liveness analysis
        fn analyze_liveness(&mut self, func: &mut Function, global_state: &GlobalState) {
            let mut state = State {
                worklist: VecDeque::from(func.analysis::<PostOrder>(global_state).reverse()),
                block_sets: HashMap::new(),
            };
            while let Some(block) = state.worklist.pop_front() {
                self.analyze_block(func, global_state, block, &mut state);
            }
        }

        /// Extend divergent liveness to the preceding uniform block. Shared memory is always
        /// uniformly declared, so it must be allocated before the branch.
        fn uniformize_liveness(&mut self, func: &mut Function, global_state: &GlobalState) {
            let mut state = State {
                worklist: VecDeque::from(func.analysis::<PostOrder>(global_state).forward()),
                block_sets: HashMap::new(),
            };
            while let Some(block) = state.worklist.pop_front() {
                self.uniformize_block(func, global_state, block, &mut state);
            }
        }

        /// Allocate slices while ensuring no concurrent shared memory slices overlap.
        /// See also [`allocate_slice`]
        fn allocate_slices(&mut self, func: &mut Function) {
            for block in func.node_ids() {
                for live_smem in self.at_block(block).clone() {
                    if !self.allocations.contains_key(&live_smem) {
                        let smem = self.shared_memories[&live_smem];
                        let offset = self.allocate_slice(block, smem.size(), smem.align);
                        self.allocations
                            .insert(smem.id, SmemAllocation { smem, offset });
                    }
                }
            }
        }

        /// Finds a valid offset for a specific slice, taking into account ranges that are already
        /// in use.
        ///
        /// Essentially the same as the global memory pool, looking for a free slice first, then
        /// extending the pool if there isn't one. Note that this linear algorithm isn't optimal
        /// for offline allocations where we know all allocations beforehand, but should be good
        /// enough for our current purposes. It may produce larger-than-required allocations in
        /// some cases. Optimal allocation would require a far more complex algorithm.
        fn allocate_slice(&mut self, block: NodeIndex, size: usize, align: usize) -> usize {
            let live_slices = self.live_slices(block);
            if live_slices.is_empty() {
                return 0;
            }

            for i in 0..live_slices.len() - 1 {
                let slice_0 = &live_slices[i];
                let slice_1 = &live_slices[i + 1];
                let end_0 = (slice_0.offset + slice_0.smem.size()).next_multiple_of(align);
                let gap = slice_1.offset.saturating_sub(end_0);
                if gap >= size {
                    return end_0;
                }
            }
            let last_slice = &live_slices[live_slices.len() - 1];
            (last_slice.offset + last_slice.smem.size()).next_multiple_of(align)
        }

        /// List of allocations that are currently live
        fn live_slices(&mut self, block: NodeIndex) -> Vec<SmemAllocation> {
            let mut live_slices = self
                .allocations
                .iter()
                .filter(|(k, _)| self.is_live(block, **k))
                .map(|it| *it.1)
                .collect::<Vec<_>>();
            live_slices.sort_by_key(|it| it.offset);
            live_slices
        }

        fn analyze_block(
            &mut self,
            func: &mut Function,
            global_state: &GlobalState,
            block: NodeIndex,
            state: &mut State,
        ) {
            let BlockSets { generated, kill } = self.block_sets(func, global_state, block, state);

            let mut live_vars = generated.clone();

            for predecessor in func.predecessors(block) {
                let predecessor = &self.live_vars[&predecessor];
                live_vars.extend(predecessor.difference(kill));
            }

            if live_vars != self.live_vars[&block] {
                state.worklist.extend(func.successors(block));
                self.live_vars.insert(block, live_vars);
            }
        }

        fn uniformize_block(
            &mut self,
            func: &mut Function,
            global_state: &GlobalState,
            block: NodeIndex,
            state: &mut State,
        ) {
            let mut live_vars = self.live_vars[&block].clone();
            let uniformity = func.analysis::<Uniformity>(global_state);

            for successor in func.successors(block) {
                if !uniformity.is_block_uniform(successor) {
                    let successor = &self.live_vars[&successor];
                    live_vars.extend(successor);
                }
            }

            if live_vars != self.live_vars[&block] {
                state.worklist.extend(func.predecessors(block));
                self.live_vars.insert(block, live_vars);
            }
        }

        fn block_sets<'a>(
            &mut self,
            func: &mut Function,
            global_state: &GlobalState,
            block: NodeIndex,
            state: &'a mut State,
        ) -> &'a BlockSets {
            let block_sets = state.block_sets.entry(block);
            block_sets.or_insert_with(|| self.calculate_block_sets(func, global_state, block))
        }

        /// Any use makes a shared memory live (`generated`), while `free` kills it (`kill`).
        /// Also collects all shared memories into a map.
        fn calculate_block_sets(
            &mut self,
            func: &mut Function,
            state: &GlobalState,
            block: NodeIndex,
        ) -> BlockSets {
            let mut generated = HashSet::new();
            let mut kill = HashSet::new();

            let ops = func[block].ops.clone();

            for op in ops.borrow_mut().values_mut() {
                func.visit_out(&mut op.out, |_, var| {
                    if let Some(smem) = shared_memory(var) {
                        generated.insert(smem.id);
                        self.shared_memories.insert(smem.id, smem);
                    }
                });
                func.visit_operation(state, &mut op.operation, |_, var| {
                    if let Some(smem) = shared_memory(var) {
                        generated.insert(smem.id);
                        self.shared_memories.insert(smem.id, smem);
                    }
                });

                if let Operation::Marker(Marker::Free(Variable {
                    kind: VariableKind::Shared { id, .. },
                    ..
                })) = &op.operation
                {
                    kill.insert(*id);
                    generated.remove(id);
                }
            }

            BlockSets { generated, kill }
        }
    }

    fn shared_memory(var: &Variable) -> Option<SharedMemory> {
        match var.kind {
            VariableKind::Shared { id, alignment } => Some(SharedMemory {
                id,
                ty: var.ty,
                align: alignment.unwrap_or_else(|| var.value_type().size()),
            }),
            _ => None,
        }
    }
}

mod captures {
    use cubecl_ir::Variable;

    use super::*;

    pub struct Captures {
        live_vars: HashMap<NodeIndex, HashSet<Variable>>,
    }

    #[derive(Clone)]
    struct BlockSets {
        generated: HashSet<Variable>,
        kill: HashSet<Variable>,
    }

    struct State {
        worklist: VecDeque<NodeIndex>,
        block_sets: HashMap<NodeIndex, BlockSets>,
    }

    impl Analysis for Captures {
        fn init(func: &mut Function, state: &GlobalState) -> Self {
            let mut this = Self::empty(func);
            this.analyze_liveness(func, state);
            this
        }
    }

    impl Captures {
        pub fn empty(func: &Function) -> Self {
            let live_vars = func
                .node_ids()
                .iter()
                .map(|it| (*it, HashSet::new()))
                .collect();
            Self { live_vars }
        }

        pub fn at_block(&self, block: NodeIndex) -> &HashSet<Variable> {
            &self.live_vars[&block]
        }

        /// Do a conservative block level liveness analysis
        pub fn analyze_liveness(&mut self, func: &mut Function, global_state: &GlobalState) {
            let mut state = State {
                worklist: VecDeque::from(func.analysis::<PostOrder>(global_state).forward()),
                block_sets: HashMap::new(),
            };
            while let Some(block) = state.worklist.pop_front() {
                self.analyze_block(func, global_state, block, &mut state);
            }
        }

        fn analyze_block(
            &mut self,
            func: &mut Function,
            global_state: &GlobalState,
            block: NodeIndex,
            state: &mut State,
        ) {
            let BlockSets { generated, kill } = block_sets(func, global_state, block, state);

            let mut live_vars = generated.clone();

            for successor in func.successors(block) {
                let successor = &self.live_vars[&successor];
                live_vars.extend(successor.difference(kill));
            }

            if live_vars != self.live_vars[&block] {
                state.worklist.extend(func.predecessors(block));
                self.live_vars.insert(block, live_vars);
            }
        }
    }

    fn block_sets<'a>(
        func: &mut Function,
        global_state: &GlobalState,
        block: NodeIndex,
        state: &'a mut State,
    ) -> &'a BlockSets {
        let block_sets = state.block_sets.entry(block);
        block_sets.or_insert_with(|| calculate_block_sets(func, global_state, block))
    }

    fn calculate_block_sets(
        func: &mut Function,
        state: &GlobalState,
        block: NodeIndex,
    ) -> BlockSets {
        let mut generated = HashSet::new();
        let mut kill = HashSet::new();

        let ops = func[block].ops.clone();

        let control_flow = func[block].control_flow.clone();
        func.visit_control_flow(&mut control_flow.borrow_mut(), |_, var| {
            generated.insert(*var);
        });
        for inst in ops.borrow_mut().values_mut().rev() {
            // Reads must be tracked after writes
            func.visit_instruction_write(state, inst, |_, var| {
                kill.insert(*var);
                generated.remove(var);
            });
            func.visit_operation(state, &mut inst.operation, |_, var| {
                generated.insert(*var);
            });
        }

        BlockSets { generated, kill }
    }
}

pub use captures::Captures;

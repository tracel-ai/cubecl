use std::collections::{HashMap, HashSet, VecDeque};

use cubecl_ir::Id;
use petgraph::graph::NodeIndex;

use crate::{Optimizer, analyses::post_order::PostOrder};

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
    fn init(opt: &mut Optimizer) -> Self {
        let mut this = Self::empty(opt);
        this.analyze_liveness(opt);
        this
    }
}

impl Liveness {
    pub fn empty(opt: &Optimizer) -> Self {
        let live_vars = opt
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
    pub fn analyze_liveness(&mut self, opt: &mut Optimizer) {
        let mut state = State {
            worklist: VecDeque::from(opt.analysis::<PostOrder>().forward()),
            block_sets: HashMap::new(),
        };
        while let Some(block) = state.worklist.pop_front() {
            self.analyze_block(opt, block, &mut state);
        }
    }

    fn analyze_block(&mut self, opt: &mut Optimizer, block: NodeIndex, state: &mut State) {
        let BlockSets { generated, kill } = block_sets(opt, block, state);

        let mut live_vars = generated.clone();

        for successor in opt.successors(block) {
            let successor = &self.live_vars[&successor];
            live_vars.extend(successor.difference(kill));
        }

        if live_vars != self.live_vars[&block] {
            state.worklist.extend(opt.predecessors(block));
            self.live_vars.insert(block, live_vars);
        }
    }
}

fn block_sets<'a>(opt: &mut Optimizer, block: NodeIndex, state: &'a mut State) -> &'a BlockSets {
    let block_sets = state.block_sets.entry(block);
    block_sets.or_insert_with(|| calculate_block_sets(opt, block))
}

fn calculate_block_sets(opt: &mut Optimizer, block: NodeIndex) -> BlockSets {
    let mut generated = HashSet::new();
    let mut kill = HashSet::new();

    let ops = opt.program[block].ops.clone();

    for op in ops.borrow_mut().values_mut().rev() {
        // Reads must be tracked after writes
        opt.visit_out(&mut op.out, |opt, var| {
            if let Some(id) = opt.local_variable_id(var) {
                kill.insert(id);
                generated.remove(&id);
            }
        });
        opt.visit_operation(&mut op.operation, &mut op.out, |opt, var| {
            if let Some(id) = opt.local_variable_id(var) {
                generated.insert(id);
            }
        });
    }

    BlockSets { generated, kill }
}

/// Shared memory liveness analysis and allocation
pub mod shared {
    use cubecl_ir::{Operation, Type, Variable, VariableKind};

    use crate::Uniformity;

    use super::*;

    /// A shared memory instance, all the information contained in the `VariableKind`, but with
    /// a non-optional `align`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SharedMemory {
        pub id: Id,
        pub length: u32,
        pub ty: Type,
        pub align: u32,
    }

    impl SharedMemory {
        /// The byte size of this shared memory
        pub fn size(&self) -> u32 {
            self.length * self.ty.size() as u32
        }
    }

    /// A specific allocation of shared memory at some `offset`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct SmemAllocation {
        /// The shared memory being allocated
        pub smem: SharedMemory,
        /// The offset in the shared memory buffer
        pub offset: u32,
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
        fn init(opt: &mut Optimizer) -> Self {
            let mut this = Self::empty(opt);
            this.analyze_liveness(opt);
            this.uniformize_liveness(opt);
            this.allocate_slices(opt);
            this
        }
    }

    impl SharedLiveness {
        pub fn empty(opt: &Optimizer) -> Self {
            let live_vars = opt
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
        fn analyze_liveness(&mut self, opt: &mut Optimizer) {
            let mut state = State {
                worklist: VecDeque::from(opt.analysis::<PostOrder>().reverse()),
                block_sets: HashMap::new(),
            };
            while let Some(block) = state.worklist.pop_front() {
                self.analyze_block(opt, block, &mut state);
            }
        }

        /// Extend divergent liveness to the preceding uniform block. Shared memory is always
        /// uniformly declared, so it must be allocated before the branch.
        fn uniformize_liveness(&mut self, opt: &mut Optimizer) {
            let mut state = State {
                worklist: VecDeque::from(opt.analysis::<PostOrder>().forward()),
                block_sets: HashMap::new(),
            };
            while let Some(block) = state.worklist.pop_front() {
                self.uniformize_block(opt, block, &mut state);
            }
        }

        /// Allocate slices while ensuring no concurrent shared memory slices overlap.
        /// See also [`allocate_slice`]
        fn allocate_slices(&mut self, opt: &mut Optimizer) {
            for block in opt.node_ids() {
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
        fn allocate_slice(&mut self, block: NodeIndex, size: u32, align: u32) -> u32 {
            let live_slices = self.live_slices(block);
            if live_slices.is_empty() {
                return 0;
            }

            for i in 0..live_slices.len() - 1 {
                let slice_0 = &live_slices[i];
                let slice_1 = &live_slices[i + 1];
                let end_0 = (slice_0.offset + slice_0.smem.size()).next_multiple_of(align);
                let gap = slice_1.offset - end_0;
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

        fn analyze_block(&mut self, opt: &mut Optimizer, block: NodeIndex, state: &mut State) {
            let BlockSets { generated, kill } = self.block_sets(opt, block, state);

            let mut live_vars = generated.clone();

            for predecessor in opt.predecessors(block) {
                let predecessor = &self.live_vars[&predecessor];
                live_vars.extend(predecessor.difference(kill));
            }

            if live_vars != self.live_vars[&block] {
                state.worklist.extend(opt.successors(block));
                self.live_vars.insert(block, live_vars);
            }
        }

        fn uniformize_block(&mut self, opt: &mut Optimizer, block: NodeIndex, state: &mut State) {
            let mut live_vars = self.live_vars[&block].clone();
            let uniformity = opt.analysis::<Uniformity>();

            for successor in opt.successors(block) {
                if !uniformity.is_block_uniform(successor) {
                    let successor = &self.live_vars[&successor];
                    live_vars.extend(successor);
                }
            }

            if live_vars != self.live_vars[&block] {
                state.worklist.extend(opt.predecessors(block));
                self.live_vars.insert(block, live_vars);
            }
        }

        fn block_sets<'a>(
            &mut self,
            opt: &mut Optimizer,
            block: NodeIndex,
            state: &'a mut State,
        ) -> &'a BlockSets {
            let block_sets = state.block_sets.entry(block);
            block_sets.or_insert_with(|| self.calculate_block_sets(opt, block))
        }

        /// Any use makes a shared memory live (`generated`), while `free` kills it (`kill`).
        /// Also collects all shared memories into a map.
        fn calculate_block_sets(&mut self, opt: &mut Optimizer, block: NodeIndex) -> BlockSets {
            let mut generated = HashSet::new();
            let mut kill = HashSet::new();

            let ops = opt.program[block].ops.clone();

            for op in ops.borrow_mut().values_mut() {
                opt.visit_out(&mut op.out, |_, var| {
                    if let Some(smem) = shared_memory(var) {
                        generated.insert(smem.id);
                        self.shared_memories.insert(smem.id, smem);
                    }
                });
                opt.visit_operation(&mut op.operation, &mut op.out, |_, var| {
                    if let Some(smem) = shared_memory(var) {
                        generated.insert(smem.id);
                        self.shared_memories.insert(smem.id, smem);
                    }
                });

                if let Operation::Free(Variable {
                    kind: VariableKind::SharedMemory { id, .. },
                    ..
                }) = &op.operation
                {
                    kill.insert(*id);
                    generated.remove(id);
                }
            }

            BlockSets { generated, kill }
        }
    }

    fn shared_memory(var: &Variable) -> Option<SharedMemory> {
        if let Variable {
            kind:
                VariableKind::SharedMemory {
                    id,
                    length,
                    unroll_factor,
                    alignment,
                },
            ..
        } = *var
        {
            Some(SharedMemory {
                id,
                length: length * unroll_factor,
                ty: var.ty,
                align: alignment.unwrap_or_else(|| var.ty.size() as u32),
            })
        } else {
            None
        }
    }
}

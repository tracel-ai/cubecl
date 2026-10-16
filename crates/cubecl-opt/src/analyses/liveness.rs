use hashbrown::HashMap;

/// Shared memory liveness analysis and allocation
pub mod shared {
    use core::{any::type_name, iter::successors};

    use alloc::vec::Vec;
    use cubecl_ir::{
        AddressSpace,
        dialect::{
            branch::{RangeLoopOp, WhileOp},
            general::FreeOp,
            memory::DeclareVariableOp,
            scf::{ForOp, WhileOp as ScfWhileOp},
            synchronization::{SyncOp, SyncScope},
        },
        interfaces::aliasing::PointerExt,
        prelude::{Context, OneResultInterface, Operation, Ptr, Result},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::attr_interfaces::TypedAttrInterface,
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op,
        },
        pass::{Analysis, AnalysisManager},
        r#type::TypeHandle,
        value::Value,
    };

    use crate::MemoryResource;

    use super::*;

    /// A specific allocation of shared memory at some `offset`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct SmemAllocation {
        pub value: Value,
        /// The type of the value (not wrapped in a pointer)
        pub value_ty: TypeHandle,
        /// The shared memory being allocated
        pub smem: MemoryResource,
        /// The offset in the shared memory buffer
        pub offset: usize,
    }

    impl SmemAllocation {
        pub fn end(&self, ctx: &Context) -> usize {
            self.offset + self.smem.size(ctx)
        }
    }

    /// Shared liveness works the other way around from normal liveness, since shared memory lives
    /// forever by default. So any use (read or write) inserts it as live, while only `free` changes
    /// the state to dead.
    ///
    /// A `free` releases its memory to a later one only where every unit of the cube is done with
    /// it before any touches the later one: the `free` and a cube barrier after it each run on
    /// every path to the later memory's first use, and run once. Each sits directly in a block
    /// enclosing that use, and no loop encloses that block. A `free` inside a branch might not
    /// run, and one inside a loop leaves the memory to be used again on the next iteration; without
    /// the barrier, one unit could write the new memory while another still reads the freed one.
    ///
    /// It also handles allocation of slices to each shared memory object, using the analyzed
    /// liveness. `allocations` contains a specific slice allocation for each shared memory, while
    /// ensuring no shared memories that exist at the same time can overlap.
    #[derive(Default, Clone)]
    pub struct SharedLiveness {
        /// Map of all shared memories by their ID. Populated during the first pass with all
        /// accessed shared memories.
        pub shared_memories: HashMap<Value, MemoryResource>,
        /// Map of allocations for each shared memory by its ID. Populated after the analysis, and
        /// should contain all memories from `shared_memories`.
        pub allocations: HashMap<Value, SmemAllocation>,
    }

    impl Analysis for SharedLiveness {
        fn name(&self) -> &str {
            type_name::<Self>()
        }

        fn compute(
            op: Ptr<Operation>,
            ctx: &Context,
            _analyses: &mut AnalysisManager,
        ) -> Result<Self>
        where
            Self: Sized,
        {
            let mut walk = Walk::default();
            walk_op(
                ctx,
                &mut walk,
                &WALKCONFIG_PREORDER_FORWARD,
                op,
                |ctx, walk, node| {
                    if let IRNode::Operation(op) = node {
                        walk.visit(ctx, op);
                    }
                },
            );
            let mut state = Self {
                shared_memories: walk.shared_memories.iter().copied().collect(),
                allocations: HashMap::new(),
            };
            for &(root_ptr, smem) in &walk.shared_memories {
                let released = walk.released_to(root_ptr);
                let offset = state.allocate_slice(ctx, smem.size(ctx), smem.alignment, &released);
                state.allocations.insert(
                    root_ptr,
                    SmemAllocation {
                        value: root_ptr,
                        value_ty: smem.value_ty,
                        smem,
                        offset,
                    },
                );
            }
            Ok(state)
        }
    }

    impl SharedLiveness {
        /// Finds a valid offset for a specific slice, taking into account ranges that are already
        /// in use.
        ///
        /// Essentially the same as the global memory pool, looking for a free slice first, then
        /// extending the pool if there isn't one. Note that this linear algorithm isn't optimal
        /// for offline allocations where we know all allocations beforehand, but should be good
        /// enough for our current purposes. It may produce larger-than-required allocations in
        /// some cases. Optimal allocation would require a far more complex algorithm.
        fn allocate_slice(
            &mut self,
            ctx: &Context,
            size: usize,
            align: usize,
            released: &[Value],
        ) -> usize {
            let mut live_slices = self
                .allocations
                .values()
                .filter(|it| !released.contains(&it.value))
                .collect::<Vec<_>>();
            live_slices.sort_by_key(|it| it.offset);
            if live_slices.is_empty() {
                return 0;
            }

            for i in 0..live_slices.len() - 1 {
                let slice_0 = &live_slices[i];
                let slice_1 = &live_slices[i + 1];
                let end_0 = (slice_0.offset + slice_0.smem.size(ctx)).next_multiple_of(align);
                let gap = slice_1.offset.saturating_sub(end_0);
                if gap >= size {
                    return end_0;
                }
            }
            let last_slice = &live_slices[live_slices.len() - 1];
            (last_slice.offset + last_slice.smem.size(ctx)).next_multiple_of(align)
        }
    }

    /// Where an op sits: its block, and its place in the walk.
    struct Place {
        block: Ptr<BasicBlock>,
        at: usize,
    }

    /// What one walk of a kernel finds: its shared memories in the order they are declared, where
    /// each is first used, and the frees and cube barriers that run once.
    #[derive(Default)]
    struct Walk {
        shared_memories: Vec<(Value, MemoryResource)>,
        first_uses: HashMap<Value, (usize, Vec<Ptr<BasicBlock>>)>,
        frees: Vec<(Value, Place)>,
        barriers: Vec<Place>,
        ops: usize,
    }

    impl Walk {
        fn visit(&mut self, ctx: &Context, op: Ptr<Operation>) {
            self.ops += 1;
            let at = self.ops;
            let op_dyn = Operation::get_op_dyn(op, ctx);
            if let Some(declare) = op_dyn.downcast_ref::<DeclareVariableOp>()
                && declare.addr_space(ctx).0 == AddressSpace::Shared
            {
                let root_ptr = declare.get_result(ctx);
                let smem = MemoryResource {
                    address_space: AddressSpace::Shared,
                    value_ty: declare.value_ty(ctx).get_type(ctx),
                    alignment: declare.alignment(ctx).0,
                    root_ptr,
                };
                self.shared_memories.push((root_ptr, smem));
                return;
            }
            let parent = op.deref(ctx).get_parent_block();
            for operand in op.deref(ctx).operands() {
                let root = operand.get_root_value(ctx);
                if !self.first_uses.contains_key(&root)
                    && self
                        .shared_memories
                        .iter()
                        .any(|&(declared, _)| declared == root)
                {
                    let enclosing = parent
                        .into_iter()
                        .flat_map(|block| enclosing_blocks(ctx, block))
                        .collect();
                    self.first_uses.insert(root, (at, enclosing));
                }
            }
            let runs_once = || parent.filter(|&block| !in_loop(ctx, block));
            if Operation::is_op::<FreeOp>(op, ctx)
                && let Some(block) = runs_once()
            {
                let memory = op.deref(ctx).get_operand(0).get_root_value(ctx);
                self.frees.push((memory, Place { block, at }));
            }
            if let Some(sync) = op_dyn.downcast_ref::<SyncOp>()
                && sync.scope(ctx).0 >= SyncScope::Cube
                && let Some(block) = runs_once()
            {
                self.barriers.push(Place { block, at });
            }
        }

        /// The memories freed for `root`: each freed, then a cube barrier passed, on every path to
        /// its first use, each once. Empty for a memory never used.
        fn released_to(&self, root: Value) -> Vec<Value> {
            let Some((used, enclosing)) = self.first_uses.get(&root) else {
                return Vec::new();
            };
            let runs_before_use =
                |place: &Place| enclosing.contains(&place.block) && place.at < *used;
            self.frees
                .iter()
                .filter(|(_, free)| {
                    runs_before_use(free)
                        && self
                            .barriers
                            .iter()
                            .any(|barrier| runs_before_use(barrier) && barrier.at > free.at)
                })
                .map(|&(memory, _)| memory)
                .collect()
        }
    }

    /// `block` and the blocks enclosing it, innermost first.
    fn enclosing_blocks(
        ctx: &Context,
        block: Ptr<BasicBlock>,
    ) -> impl Iterator<Item = Ptr<BasicBlock>> + '_ {
        successors(Some(block), |block| {
            block
                .deref(ctx)
                .get_parent_op(ctx)
                .and_then(|op| op.deref(ctx).get_parent_block())
        })
    }

    /// Whether a loop encloses `block`, so that what runs in it may run more than once.
    fn in_loop(ctx: &Context, block: Ptr<BasicBlock>) -> bool {
        enclosing_blocks(ctx, block)
            .filter_map(|block| block.deref(ctx).get_parent_op(ctx))
            .any(|op| {
                Operation::is_op::<RangeLoopOp>(op, ctx)
                    || Operation::is_op::<WhileOp>(op, ctx)
                    || Operation::is_op::<ForOp>(op, ctx)
                    || Operation::is_op::<ScfWhileOp>(op, ctx)
            })
    }
}

use hashbrown::HashMap;

/// Shared memory liveness analysis and allocation
pub mod shared {
    use core::any::type_name;

    use alloc::vec::Vec;
    use cubecl_ir::{
        AddressSpace,
        dialect::memory::DeclareVariableOp,
        prelude::{Context, OneResultInterface, Operation, Ptr, Result},
    };
    use pliron::{
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
            let mut state = Self::default();
            walk_op(
                ctx,
                &mut state,
                &WALKCONFIG_PREORDER_FORWARD,
                op,
                |ctx, state, node| {
                    if let IRNode::Operation(op) = node {
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
                            state.shared_memories.insert(root_ptr, smem);
                            if !state.allocations.contains_key(&root_ptr) {
                                let offset =
                                    state.allocate_slice(ctx, smem.size(ctx), smem.alignment);
                                state.allocations.insert(
                                    root_ptr,
                                    SmemAllocation {
                                        value: root_ptr,
                                        value_ty: declare.value_ty(ctx).get_type(ctx),
                                        smem,
                                        offset,
                                    },
                                );
                            }
                        }
                    }
                },
            );
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
        fn allocate_slice(&mut self, ctx: &Context, size: usize, align: usize) -> usize {
            let mut live_slices = self.allocations.values().collect::<Vec<_>>();
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
}

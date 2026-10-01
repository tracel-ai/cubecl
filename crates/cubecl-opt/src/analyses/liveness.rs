use hashbrown::HashMap;

/// Shared memory liveness analysis and allocation
pub mod shared {
    use core::any::type_name;

    use alloc::vec::Vec;
    use cubecl_ir::{
        AddressSpace,
        dialect::{general::FreeOp, memory::DeclareVariableOp},
        prelude::{Context, OneResultInterface, Operation, Ptr, Result},
    };
    use hashbrown::HashSet;
    use pliron::{
        builtin::attr_interfaces::TypedAttrInterface,
        graph::walkers::{
            IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op,
        },
        pass::{Analysis, AnalysisManager},
        r#type::TypeHandle,
        value::{DefiningEntity, Value},
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
    /// forever by default. A declaration makes it live, and only `free` makes it dead: a shared
    /// memory declared after another is freed, in the walk's order, may take the freed bytes.
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
        /// The shared memories declared and not yet freed at the point the walk has reached: the
        /// allocations a new declaration may not overlap.
        live: HashSet<Value>,
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
                        if op_dyn.downcast_ref::<FreeOp>().is_some() {
                            let memory = op.deref(ctx).get_operand(0);
                            if let Some(root) = state.declaration_of(ctx, memory) {
                                state.live.remove(&root);
                            }
                            return;
                        }
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
                                state.live.insert(root_ptr);
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
        /// The shared memory declaration `memory` points into: the one this analysis has seen
        /// that its defining operations' operands lead back to, through whatever slicing, field
        /// extraction or casting derived it. `None` where it leads to none, or to more than one.
        fn declaration_of(&self, ctx: &Context, memory: Value) -> Option<Value> {
            let mut found = None;
            let mut seen = HashSet::new();
            let mut pending = alloc::vec![memory];
            while let Some(value) = pending.pop() {
                if !seen.insert(value) {
                    continue;
                }
                if self.shared_memories.contains_key(&value) {
                    match found {
                        None => found = Some(value),
                        Some(other) if other != value => return None,
                        Some(_) => {}
                    }
                    continue;
                }
                if let DefiningEntity::Op(op) = value.defining_entity() {
                    pending.extend(op.deref(ctx).operands());
                }
            }
            found
        }

        /// Finds a valid offset for a specific slice, taking into account ranges that are already
        /// in use.
        ///
        /// Essentially the same as the global memory pool, looking for a free slice first, then
        /// extending the pool if there isn't one. Note that this linear algorithm isn't optimal
        /// for offline allocations where we know all allocations beforehand, but should be good
        /// enough for our current purposes. It may produce larger-than-required allocations in
        /// some cases. Optimal allocation would require a far more complex algorithm.
        fn allocate_slice(&mut self, ctx: &Context, size: usize, align: usize) -> usize {
            let mut live_slices = self
                .allocations
                .values()
                .filter(|it| self.live.contains(&it.value))
                .collect::<Vec<_>>();
            live_slices.sort_by_key(|it| it.offset);
            // First fit over the gaps between live slices, the one before the first included:
            // freed bytes may sit anywhere, the start of the block too.
            let mut end = 0usize;
            for slice in live_slices {
                let start = end.next_multiple_of(align);
                if slice.offset.saturating_sub(start) >= size {
                    return start;
                }
                end = end.max(slice.offset + slice.smem.size(ctx));
            }
            end.next_multiple_of(align)
        }
    }
}

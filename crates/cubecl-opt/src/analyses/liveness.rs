use hashbrown::HashMap;

/// Shared memory liveness analysis and allocation
pub mod shared {
    use core::any::type_name;

    use alloc::vec::Vec;
    use cubecl_ir::{
        AddressSpace,
        dialect::OperationPtrExt,
        dialect::{
            MemoryClobbers,
            asm::InlineAsmOp,
            branch::{IfOp as BranchIfOp, SwitchOp as BranchSwitchOp},
            cf::{BranchConditionalOp, BranchOp, SwitchOp as CfSwitchOp},
            memory::DeclareVariableOp,
            scf::{IfOp, SwitchOp},
            synchronization::{SyncOp, SyncScope},
        },
        interfaces::{MemoryEffects, aliasing::PointerExt},
        prelude::{Context, OneResultInterface, Operation, Ptr, Result},
        types::PointerType,
    };
    use pliron::{
        builtin::attr_interfaces::TypedAttrInterface,
        linked_list::ContainsLinkedList,
        op::op_cast,
        pass::{Analysis, AnalysisManager},
        r#type::{TypeHandle, Typed},
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

    /// Where each shared memory is live, and the allocation that follows from it.
    ///
    /// A shared memory is live from the first operation that touches it to the last, in program
    /// order, and an access inside a loop keeps it live for the whole loop, since the next
    /// iteration reaches the start again. Two shared memories may share bytes only where one is
    /// dead before the other is born **and** an unconditional cube barrier separates the two:
    /// liveness is per unit, and without the barrier one unit could write the second while another
    /// still reads the first.
    ///
    /// Where an access cannot be attributed — a shared pointer whose root is not a declaration
    /// (carried through a block argument or through memory), inline assembly that may touch
    /// memory, or unstructured
    /// control flow, whose back edges program order cannot see — no two shared memories share
    /// bytes, which is the allocation this analysis made before it tracked liveness at all.
    ///
    /// `allocations` contains a specific slice allocation for each shared memory, while ensuring
    /// no shared memories that exist at the same time can overlap.
    #[derive(Default, Clone)]
    pub struct SharedLiveness {
        /// Map of all shared memories by their ID. Populated during the first pass with all
        /// accessed shared memories.
        pub shared_memories: HashMap<Value, MemoryResource>,
        /// Map of allocations for each shared memory by its ID. Populated after the analysis, and
        /// should contain all memories from `shared_memories`.
        pub allocations: HashMap<Value, SmemAllocation>,
    }

    /// The first and last program-order positions a shared memory is touched at.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Interval {
        first: usize,
        last: usize,
    }

    impl Interval {
        fn at(position: usize) -> Self {
            Self {
                first: position,
                last: position,
            }
        }

        fn cover(&mut self, other: Interval) {
            self.first = self.first.min(other.first);
            self.last = self.last.max(other.last);
        }
    }

    /// What one walk of a kernel finds: its shared memories in declaration order, the positions
    /// each is touched at, the barriers every unit reaches, and whether anything escaped.
    #[derive(Default)]
    struct Walk {
        next: usize,
        declarations: Vec<MemoryResource>,
        /// Accesses outside every loop, by position.
        live: HashMap<Value, Interval>,
        /// Accesses inside a loop, by the outermost loop's span: resolved once the loop is walked.
        in_loop: Vec<(Value, usize)>,
        /// The span of every outermost loop, by the position of the loop operation.
        loops: HashMap<usize, Interval>,
        /// Positions of cube barriers outside any branch or loop.
        barriers: Vec<usize>,
        /// Set where an access cannot be attributed to a shared memory.
        opaque: bool,
    }

    impl Walk {
        fn operation(
            &mut self,
            ctx: &Context,
            op: Ptr<Operation>,
            loop_at: Option<usize>,
            guarded: bool,
        ) {
            let position = self.next;
            self.next += 1;
            let op_dyn = Operation::get_op_dyn(op, ctx);

            let declares_shared = op_dyn
                .downcast_ref::<DeclareVariableOp>()
                .is_some_and(|declare| declare.addr_space(ctx).0 == AddressSpace::Shared);
            if let Some(declare) = op_dyn.downcast_ref::<DeclareVariableOp>()
                && declares_shared
            {
                self.declarations.push(MemoryResource {
                    address_space: AddressSpace::Shared,
                    value_ty: TypedAttrInterface::get_type(&*declare.value_ty(ctx), ctx),
                    alignment: declare.alignment(ctx).0,
                    root_ptr: declare.get_result(ctx),
                });
            }
            let asm_reaches_memory = op_dyn
                .downcast_ref::<InlineAsmOp>()
                .is_some_and(|asm| *asm.memory_clobbers(ctx) != MemoryClobbers::Nomem);
            if asm_reaches_memory
                || op_dyn.is::<BranchOp>()
                || op_dyn.is::<BranchConditionalOp>()
                || op_dyn.is::<CfSwitchOp>()
            {
                self.opaque = true;
            }
            if let Some(sync) = op_dyn.downcast_ref::<SyncOp>()
                && sync.scope(ctx).0 >= SyncScope::Cube
                && !guarded
            {
                self.barriers.push(position);
            }

            // Only an operation that reads or writes memory touches it: a memory is born where it
            // is first read or written, through whichever pointer, and not where a pointer into it
            // is declared, derived or wrapped into a composite. What reaches memory is read the way
            // the memory SSA reads it: the effects an operation states, and every memory for one
            // that states none, since nothing then says what it touches.
            let touches_memory = match op_cast::<dyn MemoryEffects>(&*op_dyn) {
                Some(effects) => !effects.memory_effects(ctx).is_empty(),
                None => !declares_shared,
            };
            let values = match touches_memory {
                true => {
                    let operation = op.deref(ctx);
                    operation
                        .operands()
                        .chain(operation.results())
                        .collect::<Vec<_>>()
                }
                false => Vec::new(),
            };
            for value in values {
                self.access(ctx, value, position, loop_at);
            }

            // Every operation with regions is control flow, and a loop unless it is known to run
            // its regions at most once: an unknown one costs reuse, never correctness. The kernel
            // itself is the walk's root and holds everything, so it is neither.
            let root = self.next == 1;
            let has_regions = op.deref(ctx).num_regions() > 0;
            let branches_once = op_dyn.is::<IfOp>()
                || op_dyn.is::<SwitchOp>()
                || op_dyn.is::<BranchIfOp>()
                || op_dyn.is::<BranchSwitchOp>();
            let is_loop = !root && has_regions && !branches_once;
            let branches = !root && has_regions;
            // Only the outermost loop matters: an inner one lies inside its span.
            let inner_loop = match (loop_at, is_loop) {
                (Some(outer), _) => Some(outer),
                (None, true) => Some(position),
                (None, false) => None,
            };
            for region in op.regions(ctx) {
                for block in region.deref(ctx).iter(ctx) {
                    for argument in block.deref(ctx).arguments() {
                        self.access(ctx, argument, self.next, inner_loop);
                    }
                    for nested in block.deref(ctx).iter(ctx) {
                        self.operation(ctx, nested, inner_loop, guarded || branches);
                    }
                }
            }
            if is_loop && loop_at.is_none() {
                self.loops.insert(
                    position,
                    Interval {
                        first: position,
                        last: self.next - 1,
                    },
                );
            }
        }

        /// Record `value` as an access at `position` where it points into shared memory.
        fn access(&mut self, ctx: &Context, value: Value, position: usize, loop_at: Option<usize>) {
            let ty = value.get_type(ctx).deref(ctx);
            let Some(pointer) = ty.downcast_ref::<PointerType>() else {
                return;
            };
            if pointer.address_space != AddressSpace::Shared {
                return;
            }
            let root = value.get_root_value(ctx);
            let declared = value
                .get_root_defining_op(ctx)
                .is_some_and(|op| Operation::get_op_dyn(op, ctx).is::<DeclareVariableOp>());
            if !declared {
                self.opaque = true;
                return;
            }
            match loop_at {
                Some(loop_position) => self.in_loop.push((root, loop_position)),
                None => {
                    self.live
                        .entry(root)
                        .and_modify(|interval| interval.cover(Interval::at(position)))
                        .or_insert(Interval::at(position));
                }
            }
        }

        /// Each shared memory's interval, its loop accesses widened to their loops.
        fn intervals(&self) -> HashMap<Value, Interval> {
            let mut live = self.live.clone();
            for &(root, loop_position) in &self.in_loop {
                let span = self.loops[&loop_position];
                live.entry(root)
                    .and_modify(|interval| interval.cover(span))
                    .or_insert(span);
            }
            live
        }

        /// Whether `a` and `b` may share bytes: one is dead before the other is born, with a
        /// barrier every unit reaches between the two.
        fn separable(&self, a: Option<Interval>, b: Option<Interval>) -> bool {
            if self.opaque {
                return false;
            }
            let (Some(a), Some(b)) = (a, b) else {
                // A shared memory nothing touches holds no value to protect.
                return true;
            };
            let (before, after) = match a.last < b.first {
                true => (a, b),
                false => (b, a),
            };
            before.last < after.first
                && self
                    .barriers
                    .iter()
                    .any(|&barrier| before.last < barrier && barrier < after.first)
        }
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
            walk.operation(ctx, op, None, false);
            let intervals = walk.intervals();

            // Placed in order of first access, so a memory born after a barrier finds the bytes
            // the memories that died before it left.
            let mut order = walk.declarations.clone();
            order.sort_by_key(|smem| {
                intervals
                    .get(&smem.root_ptr)
                    .map_or(usize::MAX, |interval| interval.first)
            });

            let mut state = Self::default();
            for smem in order {
                let root_ptr = smem.root_ptr;
                state.shared_memories.insert(root_ptr, smem);
                if state.allocations.contains_key(&root_ptr) {
                    continue;
                }
                let interval = intervals.get(&root_ptr).copied();
                let conflicting = state
                    .allocations
                    .values()
                    .filter(|placed| {
                        !walk.separable(interval, intervals.get(&placed.value).copied())
                    })
                    .copied()
                    .collect::<Vec<_>>();
                let offset = first_fit(ctx, &conflicting, smem.size(ctx), smem.alignment);
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

    /// The lowest offset aligned to `align` where `size` bytes overlap none of `taken`.
    fn first_fit(ctx: &Context, taken: &[SmemAllocation], size: usize, align: usize) -> usize {
        let mut taken = taken.to_vec();
        taken.sort_by_key(|slice| slice.offset);
        let mut offset = 0usize;
        for slice in taken {
            if offset + size <= slice.offset {
                return offset;
            }
            offset = offset.max(slice.end(ctx).next_multiple_of(align));
        }
        offset
    }
}

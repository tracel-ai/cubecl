use hashbrown::HashMap;

/// Shared memory liveness analysis and allocation
pub mod shared {
    use core::any::type_name;

    use alloc::vec::Vec;
    use cubecl_ir::{
        AddressSpace,
        dialect::OperationPtrExt,
        dialect::{
            barrier::{CopyAsyncOp, MemCopyAsyncOp, MemCopyAsyncTxOp},
            branch::{ReturnOp, UnreachableOp},
            memory::DeclareVariableOp,
            synchronization::SyncScope,
            tma::{TmaLoadOp, TmaStoreOp},
        },
        interfaces::{
            MemoryEffect, MemoryEffects, Synchronizes,
            aliasing::PointerExt,
            control_flow::{RegionBranchOpInterface, RegionSuccessor},
        },
        prelude::{Context, OneResultInterface, Operation, Ptr, Result},
        types::{ArrayType, PointerType, barrier::BarrierType},
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
    /// dead before the other is born **and** a cube barrier every unit reaches separates the two:
    /// liveness is per unit, and without the barrier one unit could write the second while
    /// another still reads the first. A unit that leaves the kernel early never reaches the
    /// barrier, so a barrier separates nothing an early exit may have left unordered.
    ///
    /// Some bytes outlive the operation that touches them. An asynchronous copy or TMA transfer
    /// reads or writes its shared memory after the operation that starts it, and no cube barrier
    /// waits for it, so a memory one touches stays live to the end of the kernel. A barrier
    /// object's bytes are never repurposed, since nothing invalidates it first.
    ///
    /// Where an access cannot be attributed to one shared memory — a shared pointer whose root is
    /// not a declaration (carried through a block argument, a `select` or memory), an operation
    /// whose effects reach memory no pointer names, or unstructured control flow, whose back
    /// edges program order cannot see — no two shared memories share bytes.
    ///
    /// `allocations` holds one slice per shared memory, no two of which overlap where both can be
    /// live at once.
    #[derive(Default, Clone)]
    pub struct SharedLiveness {
        /// Every shared memory the kernel declares, by its root pointer.
        pub shared_memories: HashMap<Value, MemoryResource>,
        /// The slice each shared memory is placed at, by its root pointer: one per entry of
        /// `shared_memories`.
        pub allocations: HashMap<Value, SmemAllocation>,
    }

    /// A place in the kernel's program order: operations are numbered in the order a walk of
    /// their regions visits them.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Position(usize);

    impl Position {
        /// Past every operation: where a memory an asynchronous transfer touches stays live to.
        const END: Self = Self(usize::MAX);

        fn next(self) -> Self {
            Self(self.0 + 1)
        }

        fn previous(self) -> Self {
            Self(self.0 - 1)
        }
    }

    /// The first and last positions a shared memory is live at.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Interval {
        first: Position,
        last: Position,
    }

    impl Interval {
        /// Live for the whole kernel: what a memory whose bytes are never reused holds.
        const WHOLE_KERNEL: Self = Self {
            first: Position(0),
            last: Position::END,
        };

        fn cover(&mut self, other: Interval) {
            self.first = self.first.min(other.first);
            self.last = self.last.max(other.last);
        }
    }

    /// Whether every unit of the cube reaches an operation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Reach {
        /// Outside every branch and loop.
        EveryUnit,
        /// Inside a branch or a loop, which units may take different ways through.
        SomeUnits,
    }

    /// Where an operation sits in the kernel.
    #[derive(Debug, Clone, Copy)]
    struct Site {
        /// The outermost loop around it, by the loop operation's position.
        outer_loop: Option<Position>,
        reach: Reach,
    }

    impl Site {
        /// The kernel's own body.
        const KERNEL: Self = Self {
            outer_loop: None,
            reach: Reach::EveryUnit,
        };

        /// The site of what lies in the regions of an operation at `position`.
        fn inside(self, position: Position, structure: Structure) -> Self {
            let outer_loop = match (self.outer_loop, structure) {
                (Some(outer), _) => Some(outer),
                (None, Structure::Loop) => Some(position),
                (None, Structure::Branch) => None,
            };
            Self {
                outer_loop,
                reach: Reach::SomeUnits,
            }
        }
    }

    /// How control moves through an operation's regions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Structure {
        /// Each region runs at most once, then control leaves the operation.
        Branch,
        /// A region can be entered again, or the operation states nothing about its regions:
        /// an unknown one costs reuse, never correctness.
        Loop,
    }

    impl Structure {
        fn of(ctx: &Context, op: Ptr<Operation>) -> Self {
            let op_dyn = Operation::get_op_dyn(op, ctx);
            let Some(branch) = op_cast::<dyn RegionBranchOpInterface>(&*op_dyn) else {
                return Self::Loop;
            };
            let reenters = op.regions(ctx).into_iter().any(|region| {
                branch
                    .all_successor_regions_of_region(ctx, region)
                    .iter()
                    .any(|successor| matches!(successor, RegionSuccessor::Region(_)))
            });
            match reenters {
                true => Self::Loop,
                false => Self::Branch,
            }
        }
    }

    /// How long an access keeps the memory it touches live.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Lasting {
        /// Done when the operation is.
        Operation,
        /// Still reading or writing after the operation, until nothing this analysis can see.
        KernelEnd,
    }

    /// What one operation is to the liveness of shared memory.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Role {
        /// Declares a shared memory: not an access.
        Declaration,
        /// Waits for every unit of the cube.
        Barrier,
        /// Ends the kernel for the units that reach it.
        Exit,
        /// Reaches memory no pointer names, or branches to blocks program order cannot follow.
        Unattributable,
        /// Reads or writes the shared memories its pointer operands and results point into, for
        /// as long as it lasts.
        Access(Lasting),
        /// Touches no memory.
        Inert,
    }

    impl Role {
        fn of(ctx: &Context, op: Ptr<Operation>) -> Self {
            let op_dyn = Operation::get_op_dyn(op, ctx);
            if let Some(declare) = op_dyn.downcast_ref::<DeclareVariableOp>() {
                return match declare.addr_space(ctx).0 {
                    AddressSpace::Shared => Self::Declaration,
                    _ => Self::Inert,
                };
            }
            if op.deref(ctx).successors().next().is_some() {
                return Self::Unattributable;
            }
            if op_cast::<dyn Synchronizes>(&*op_dyn)
                .is_some_and(|sync| sync.minimum_scope(ctx) >= SyncScope::Cube)
            {
                return Self::Barrier;
            }
            if op_dyn.is::<ReturnOp>() || op_dyn.is::<UnreachableOp>() {
                return Self::Exit;
            }
            let lasting = match op_dyn.is::<CopyAsyncOp>()
                || op_dyn.is::<MemCopyAsyncOp>()
                || op_dyn.is::<MemCopyAsyncTxOp>()
                || op_dyn.is::<TmaLoadOp>()
                || op_dyn.is::<TmaStoreOp>()
            {
                true => Lasting::KernelEnd,
                false => Lasting::Operation,
            };
            // What reaches memory is read the way the memory SSA reads it: the effects an
            // operation states, and every memory for one that states none, since nothing then
            // says what it touches.
            let Some(effects) = op_cast::<dyn MemoryEffects>(&*op_dyn) else {
                return Self::Access(lasting);
            };
            let effects = effects.memory_effects(ctx);
            if effects.iter().any(reaches_unnamed_shared_memory) {
                return Self::Unattributable;
            }
            match effects.is_empty() {
                true => Self::Inert,
                false => Self::Access(lasting),
            }
        }
    }

    /// Whether `effect` may reach shared memory through no pointer it names.
    fn reaches_unnamed_shared_memory(effect: &MemoryEffect) -> bool {
        match effect {
            MemoryEffect::Read(_) | MemoryEffect::Write(_) => false,
            MemoryEffect::ReadAllInSpace(space) | MemoryEffect::WriteAllInSpace(space) => {
                *space == AddressSpace::Shared
            }
            MemoryEffect::ReadAll | MemoryEffect::WriteAll | MemoryEffect::Opaque => true,
        }
    }

    /// Whether every access to shared memory names the memory it touches.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    enum Attribution {
        #[default]
        Tracked,
        /// Some access cannot be tied to one memory: none may share bytes.
        Unattributable,
    }

    /// The kernel's shared memories and every access, barrier and exit that bears on them, in
    /// program order.
    #[derive(Default)]
    struct SharedAccesses {
        next: Position,
        declarations: Vec<MemoryResource>,
        /// Accesses outside every loop, and every access that outlasts its operation.
        live: HashMap<Value, Interval>,
        /// Accesses inside a loop, by the outermost loop's position: resolved once it is walked.
        in_loop: Vec<(Value, Position)>,
        /// The span of every outermost loop, by the loop operation's position.
        loops: HashMap<Position, Interval>,
        /// Cube barriers every unit reaches.
        barriers: Vec<Position>,
        /// Operations that end the kernel for the units reaching them.
        exits: Vec<Position>,
        attribution: Attribution,
    }

    impl SharedAccesses {
        /// Walk `kernel`'s body.
        fn of(ctx: &Context, kernel: Ptr<Operation>) -> Self {
            let mut accesses = Self::default();
            accesses.regions(ctx, kernel, Site::KERNEL);
            accesses
        }

        fn operation(&mut self, ctx: &Context, op: Ptr<Operation>, site: Site) {
            let position = self.next;
            self.next = position.next();
            match Role::of(ctx, op) {
                Role::Declaration => self.declare(ctx, op),
                Role::Barrier => self.barrier(position, site),
                Role::Exit => self.exits.push(position),
                Role::Unattributable => self.attribution = Attribution::Unattributable,
                Role::Access(lasting) => self.touch(ctx, op, position, site, lasting),
                Role::Inert => {}
            }
            if op.deref(ctx).num_regions() > 0 {
                self.nested(ctx, op, position, site);
            }
        }

        /// The regions of the operation at `position`, and the span it covers if it is an
        /// outermost loop.
        fn nested(&mut self, ctx: &Context, op: Ptr<Operation>, position: Position, site: Site) {
            let structure = Structure::of(ctx, op);
            self.regions(ctx, op, site.inside(position, structure));
            if structure == Structure::Loop && site.outer_loop.is_none() {
                let span = Interval {
                    first: position,
                    last: self.next.previous(),
                };
                self.loops.insert(position, span);
            }
        }

        fn regions(&mut self, ctx: &Context, op: Ptr<Operation>, site: Site) {
            for region in op.regions(ctx) {
                for block in region.deref(ctx).iter(ctx) {
                    for argument in block.deref(ctx).arguments() {
                        self.access(ctx, argument, self.next, site, Lasting::Operation);
                    }
                    for nested in block.deref(ctx).iter(ctx) {
                        self.operation(ctx, nested, site);
                    }
                }
            }
        }

        fn declare(&mut self, ctx: &Context, op: Ptr<Operation>) {
            let op_dyn = Operation::get_op_dyn(op, ctx);
            let declare = op_dyn
                .downcast_ref::<DeclareVariableOp>()
                .expect("a declaration's role is read off its op");
            let smem = MemoryResource {
                address_space: AddressSpace::Shared,
                value_ty: TypedAttrInterface::get_type(&*declare.value_ty(ctx), ctx),
                alignment: declare.alignment(ctx).0,
                root_ptr: declare.get_result(ctx),
            };
            if holds_barrier_objects(ctx, smem.value_ty) {
                self.cover(smem.root_ptr, Interval::WHOLE_KERNEL);
            }
            self.declarations.push(smem);
        }

        fn barrier(&mut self, position: Position, site: Site) {
            if site.reach == Reach::EveryUnit {
                self.barriers.push(position);
            }
        }

        /// Every shared memory `op`'s pointer operands and results point into.
        fn touch(
            &mut self,
            ctx: &Context,
            op: Ptr<Operation>,
            position: Position,
            site: Site,
            lasting: Lasting,
        ) {
            let operation = op.deref(ctx);
            for value in operation.operands().chain(operation.results()) {
                self.access(ctx, value, position, site, lasting);
            }
        }

        /// Record `value` as an access at `position` where it points into shared memory.
        fn access(
            &mut self,
            ctx: &Context,
            value: Value,
            position: Position,
            site: Site,
            lasting: Lasting,
        ) {
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
                self.attribution = Attribution::Unattributable;
                return;
            }
            // A loop's accesses are widened to the loop once it is walked; one that outlasts its
            // operation is live to the end wherever it sits.
            if let Some(outer_loop) = site.outer_loop {
                self.in_loop.push((root, outer_loop));
            }
            let interval = match (site.outer_loop, lasting) {
                (_, Lasting::KernelEnd) => Interval {
                    first: position,
                    last: Position::END,
                },
                (None, Lasting::Operation) => Interval {
                    first: position,
                    last: position,
                },
                (Some(_), Lasting::Operation) => return,
            };
            self.cover(root, interval);
        }

        fn cover(&mut self, root: Value, interval: Interval) {
            self.live
                .entry(root)
                .and_modify(|live| live.cover(interval))
                .or_insert(interval);
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

        /// Whether memories live over `a` and `b` may share bytes: one is dead before the other
        /// is born, with a barrier every unit reaches between the two and no exit before it that
        /// a unit could have left the first memory unordered by.
        fn separable(&self, a: Option<Interval>, b: Option<Interval>) -> bool {
            if self.attribution == Attribution::Unattributable {
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
            let exits_between = |barrier: Position| {
                self.exits
                    .iter()
                    .any(|&exit| before.first <= exit && exit < barrier)
            };
            before.last < after.first
                && self.barriers.iter().any(|&barrier| {
                    before.last < barrier && barrier < after.first && !exits_between(barrier)
                })
        }
    }

    /// Whether `ty` is a barrier object or an array of them, whose bytes nothing invalidates.
    fn holds_barrier_objects(ctx: &Context, ty: TypeHandle) -> bool {
        let ty = ty.deref(ctx);
        if ty.is::<BarrierType>() {
            return true;
        }
        ty.downcast_ref::<ArrayType>()
            .is_some_and(|array| holds_barrier_objects(ctx, array.inner))
    }

    impl SharedLiveness {
        /// The lowest offset aligned to `smem`'s alignment where it overlaps no placed memory it
        /// may not share bytes with.
        fn lowest_free_offset(
            &self,
            ctx: &Context,
            smem: &MemoryResource,
            intervals: &HashMap<Value, Interval>,
            accesses: &SharedAccesses,
        ) -> usize {
            let interval = intervals.get(&smem.root_ptr).copied();
            let mut taken = self
                .allocations
                .values()
                .filter(|placed| {
                    !accesses.separable(interval, intervals.get(&placed.value).copied())
                })
                .collect::<Vec<_>>();
            taken.sort_by_key(|slice| slice.offset);
            let size = smem.size(ctx);
            let mut offset = 0usize;
            for slice in taken {
                if offset + size <= slice.offset {
                    return offset;
                }
                offset = offset.max(slice.end(ctx).next_multiple_of(smem.alignment));
            }
            offset
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
            let accesses = SharedAccesses::of(ctx, op);
            let intervals = accesses.intervals();

            // Placed in order of first access, so a memory born after a barrier finds the bytes
            // the memories that died before it left.
            let mut order = accesses.declarations.clone();
            order.sort_by_key(|smem| {
                intervals
                    .get(&smem.root_ptr)
                    .map_or(Position::END, |interval| interval.first)
            });

            let mut state = Self::default();
            for smem in order {
                let root_ptr = smem.root_ptr;
                state.shared_memories.insert(root_ptr, smem);
                if state.allocations.contains_key(&root_ptr) {
                    continue;
                }
                let offset = state.lowest_free_offset(ctx, &smem, &intervals, &accesses);
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
}

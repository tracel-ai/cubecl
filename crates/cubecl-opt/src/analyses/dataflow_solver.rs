use core::{
    any::{TypeId, type_name},
    cell::RefCell,
    hash::{BuildHasher, Hash, Hasher},
    marker::PhantomData,
    ops::BitOrAssign,
};

use alloc::{boxed::Box, collections::VecDeque, rc::Rc, vec::Vec};
use cubecl_environment::collections::HashMap;
use cubecl_ir::prelude::*;
use downcast_rs::{Downcast, impl_downcast};
use pliron::{
    basic_block::BasicBlock,
    graph::HasLabel,
    linked_list::{ContainsLinkedList, LinkedList},
    printable::Printable,
    verify_err_noloc,
};
use rustc_hash::FxBuildHasher;

use smallvec::SmallVec;
pub use solver::*;

pub mod dead_code;
pub mod sccp;
pub mod sparse;

pub type SmallPtrVec<T> = SmallVec<[T; 8]>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeResult {
    Changed,
    Unchanged,
}

impl BitOrAssign for ChangeResult {
    fn bitor_assign(&mut self, rhs: Self) {
        if matches!(rhs, ChangeResult::Changed) {
            *self = ChangeResult::Changed;
        }
    }
}

/// Nested module to ensure none of the unsafe abstractions leave this scope.
mod solver {
    use super::*;

    pub use unsafe_zone::{DataflowSolver, ReadRef};

    pub type SolverWorkItem = (ProgramPoint, TypeId);

    /// Special care needs to be taken here to ensure invariants hold. Only functions directly using
    /// the unsafe portions of the code should be in here.
    ///
    /// # Safety invariants
    /// * State must not be removed or re-inserted through a `&self`. Only fresh insertion and
    ///   in-place mutation are allowed.
    /// * State must only be mutated after checking no readable references exist via `read_lock`
    /// * Mutable references to state created via `&self` must not be leaked outside this module
    mod unsafe_zone {
        use core::cell::{Cell, Ref, RefMut};

        use derive_more::Deref;

        use super::*;

        /// Read-only ref, required for soundness to ensure read-only lattices are never borrowed at the
        /// same time as write-only lattices. Uses a pointer to store the value, because the borrow
        /// lives as long as the solver, not as long as the reference to the map. This is safe because
        /// an immutable borrow to the solver can only be used to insert values, never remove them. The
        /// values are also boxed, so aren't moved when the map grows.
        /// Removing or re-inserting values must require a mutable reference to the solver, so read refs
        /// are forced to be dropped.
        pub struct ReadRef<'a, T> {
            value: *const T,
            /// Overlap is only allowed between writes, not between reads and writes.
            /// The lock ensures all readable borrows are dropped before a writing ref is created.
            read_lock: Rc<Cell<usize>>,
            _solver: PhantomData<&'a ()>,
        }

        impl<T> ReadRef<'_, T> {
            pub fn deref(&self) -> BorrowGuard<'_, T> {
                self.read_lock.update(|it| it + 1);
                BorrowGuard {
                    value: unsafe { &*self.value },
                    read_lock: &self.read_lock,
                }
            }
        }

        #[derive(Deref)]
        pub struct BorrowGuard<'a, T> {
            #[deref]
            value: &'a T,
            read_lock: &'a Cell<usize>,
        }

        impl<T> Drop for BorrowGuard<'_, T> {
            fn drop(&mut self) {
                self.read_lock.update(|it| it - 1);
            }
        }

        pub(super) struct StateEntry {
            read_lock: Rc<Cell<usize>>,
            value: Box<dyn PrintableState>,
        }

        impl StateEntry {
            pub fn new(value: impl PrintableState) -> Self {
                StateEntry {
                    read_lock: Default::default(),
                    value: Box::new(value),
                }
            }

            fn read<'a, T: PrintableState>(&self) -> ReadRef<'a, T> {
                ReadRef {
                    value: self.value.downcast_ref().unwrap(),
                    read_lock: self.read_lock.clone(),
                    _solver: PhantomData,
                }
            }

            pub(super) unsafe fn inner(&self) -> &dyn PrintableState {
                &*self.value
            }

            fn borrow_mut<T: PrintableState>(&mut self) -> &mut T {
                if self.read_lock.get() > 0 {
                    panic!("Can't update state while it's immutably borrowed");
                }
                self.value.downcast_mut().unwrap()
            }
        }

        type AnalysisStates = HashMap<u64, States>;
        type States = HashMap<TypeId, StateEntry>;

        pub struct DataflowSolver {
            pub(super) child_analyses: HashMap<TypeId, Box<dyn DataflowAnalysis>>,
            pub(super) worklist: RefCell<VecDeque<SolverWorkItem>>,
            pub(super) anchor_hash: FxBuildHasher,
            /// Pre-hashed anchor to typed state. Allows arbitrary anchor types.
            /// Removing or re-inserting elements *must* be done through a mutable reference to ensure
            /// no dangling read refs exist.
            analysis_states: RefCell<AnalysisStates>,
            pub(super) config: SolverConfig,
        }

        impl DataflowSolver {
            pub fn new(config: SolverConfig) -> Self {
                Self {
                    child_analyses: Default::default(),
                    worklist: Default::default(),
                    anchor_hash: Default::default(),
                    analysis_states: Default::default(),
                    config,
                }
            }

            pub fn update_state<A: AnalysisState>(
                &self,
                ctx: &Context,
                state: &WriteRef<A>,
                update: impl FnOnce(&mut A) -> ChangeResult,
            ) {
                let mut states = self.analysis_states.borrow_mut();
                let states = states.get_mut(&state.anchor).unwrap();
                let state = states.get_mut(&TypeId::of::<A>()).unwrap().borrow_mut();
                let changed = update(state);
                if changed == ChangeResult::Changed {
                    state.on_update(ctx, self);
                }
            }

            pub fn get_or_create<T: AnalysisState>(&self, anchor: T::Anchor) -> ReadRef<'_, T> {
                let anchor_hash = self.hash_anchor(&anchor);
                let mut states = self.analysis_states.borrow_mut();
                let state = states.entry(anchor_hash).or_default();
                let id = TypeId::of::<T>();
                state
                    .entry(id)
                    .or_insert_with(|| StateEntry::new(T::create(anchor)))
                    .read()
            }

            pub fn get_or_create_mut<T: AnalysisState>(
                &self,
                anchor: T::Anchor,
            ) -> WriteRef<'_, T> {
                let anchor_hash = self.hash_anchor(&anchor);
                let mut states = self.analysis_states.borrow_mut();
                let state = states.entry(anchor_hash).or_default();
                let id = TypeId::of::<T>();
                if !state.contains_key(&id) {
                    state.insert(id, StateEntry::new(T::create(anchor)));
                }
                WriteRef {
                    anchor: anchor_hash,
                    _ty: PhantomData,
                    _solver: PhantomData,
                }
            }

            pub fn get_or_create_for<A: 'static, T: AnalysisState>(
                &self,
                dependent: ProgramPoint,
                anchor: T::Anchor,
            ) -> ReadRef<'_, T> {
                let is_equivalent = self.is_equivalent::<T>(&anchor, dependent);
                let state = self.get_or_create::<T>(anchor);
                if !is_equivalent {
                    state.deref().add_dependency::<A>(dependent);
                }
                state
            }

            pub fn lookup_state<T: AnalysisState>(
                &self,
                anchor: T::Anchor,
            ) -> Option<ReadRef<'_, T>> {
                let anchor_hash = self.hash_anchor(&anchor);
                let states = self.analysis_states.borrow();
                Some(states.get(&anchor_hash)?.get(&TypeId::of::<T>())?.read())
            }

            pub(super) fn states(&self) -> Ref<'_, AnalysisStates> {
                self.analysis_states.borrow()
            }

            /// Anything not inside this module must not get a mutable borrow from an immutable ref
            #[expect(unused, reason = "For clarity in case it's needed in the future")]
            pub(super) fn states_mut(&mut self) -> RefMut<'_, AnalysisStates> {
                self.analysis_states.borrow_mut()
            }
        }
    }

    /// Write-only ref, required for soundness in cases where multiple aliasing lattice elements are
    /// referenced at the same time. Mutation is only allowed through `update_state`.
    pub struct WriteRef<'a, T> {
        anchor: u64,
        _ty: PhantomData<T>,
        _solver: PhantomData<&'a ()>,
    }

    pub struct SolverConfig {
        pub is_interprocedural: bool,
    }

    impl Default for SolverConfig {
        fn default() -> Self {
            Self {
                is_interprocedural: true,
            }
        }
    }

    impl DataflowSolver {
        pub fn load<A: DataflowAnalysis>(&mut self, analysis: A) {
            let key = TypeId::of::<A>();
            let existing = self.child_analyses.insert(key, Box::new(analysis));
            assert!(
                existing.is_none(),
                "Tried loading {} twice",
                type_name::<A>()
            )
        }

        pub fn require_loaded<A: DataflowAnalysis>(&self) -> Result<()> {
            if !self.child_analyses.contains_key(&TypeId::of::<A>()) {
                return verify_err_noloc!(
                    "Missing required dataflow analysis {}",
                    type_name::<A>()
                );
            }
            Ok(())
        }

        pub fn initialize_and_run(&mut self, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
            let is_interprocedural = self.config.is_interprocedural;
            if is_interprocedural && !root.impls::<dyn SymbolTableInterface>(ctx) {
                self.config.is_interprocedural = false;
            }

            // Take it temporarily so we get mutable access without borrowing self
            let mut child_analyses = core::mem::take(&mut self.child_analyses);

            // Initialize equivalent lattice anchors.
            for analysis in child_analyses.values() {
                analysis.initialize_equivalent_lattice_anchor(self, ctx, root);
            }

            // Initialize the analyses.
            for analysis in child_analyses.values_mut() {
                if let Err(err) = analysis.initialize(self, ctx, root) {
                    self.child_analyses = child_analyses;
                    self.config.is_interprocedural = is_interprocedural;
                    return Err(err);
                }
            }

            self.child_analyses = child_analyses;

            while let Some((point, analysis)) = {
                let mut worklist = self.worklist.borrow_mut();
                worklist.pop_front()
            } {
                let analysis = self.child_analyses.get(&analysis).expect("Should exist");
                if let Err(err) = analysis.visit(self, ctx, point) {
                    self.config.is_interprocedural = is_interprocedural;
                    return Err(err);
                }
            }

            self.config.is_interprocedural = is_interprocedural;
            Ok(())
        }

        pub fn enqueue(&self, work_item: SolverWorkItem) {
            self.worklist.borrow_mut().push_back(work_item);
        }

        pub fn config(&self) -> &SolverConfig {
            &self.config
        }

        pub fn is_equivalent<T: AnalysisState>(
            &self,
            _lhs: &T::Anchor,
            _rhs: ProgramPoint,
        ) -> bool {
            // TODO: Support equivalence classes
            false
        }

        // Includes `TypeID` so different anchor types hash to different values even if the inner data
        // is the same.
        fn hash_anchor<T: Clone + Printable + Hash + 'static>(&self, anchor: &T) -> u64 {
            let mut hasher = self.anchor_hash.build_hasher();
            TypeId::of::<T>().hash(&mut hasher);
            anchor.hash(&mut hasher);
            hasher.finish()
        }
    }

    impl Printable for DataflowSolver {
        fn fmt(
            &self,
            ctx: &Context,
            _state: &pliron::printable::State,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            writeln!(f, "DataflowSolver {{")?;
            let states = self.states();
            let mut entries = states
                .values()
                .flat_map(|states| states.iter())
                .collect::<Vec<_>>();
            entries.sort_by_key(|it| it.0);

            for (_, state) in entries {
                writeln!(f, "    {},", unsafe { state.inner() }.disp(ctx))?;
            }
            writeln!(f, "}}")
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ProgramPoint {
    Operation(Ptr<Operation>),
    BeforeOpInBlock(Ptr<BasicBlock>, Ptr<Operation>),
    EndOfBlock(Ptr<BasicBlock>),
}

impl Printable for ProgramPoint {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            ProgramPoint::Operation(op) => write!(f, "ProgramPoint::Operation({})", op.disp(ctx)),
            ProgramPoint::BeforeOpInBlock(block, _) if self.is_block_start(ctx) => {
                write!(f, "ProgramPoint::StartOfBlock({})", block.label(ctx))
            }
            ProgramPoint::BeforeOpInBlock(block, op) => {
                write!(
                    f,
                    "ProgramPoint::BeforeOpInBlock({}, {})",
                    block.label(ctx),
                    op.disp(ctx)
                )
            }
            ProgramPoint::EndOfBlock(block) => {
                write!(f, "ProgramPoint::EndOfBlock({})", block.label(ctx))
            }
        }
    }
}

impl ProgramPoint {
    pub fn before_op(ctx: &Context, op: Ptr<Operation>) -> ProgramPoint {
        if let Some(block) = op.deref(ctx).get_parent_block() {
            ProgramPoint::BeforeOpInBlock(block, op)
        } else {
            ProgramPoint::Operation(op)
        }
    }

    pub fn at_block_start(ctx: &Context, block: Ptr<BasicBlock>) -> ProgramPoint {
        if let Some(first) = block.deref(ctx).iter(ctx).next() {
            ProgramPoint::BeforeOpInBlock(block, first)
        } else {
            ProgramPoint::EndOfBlock(block)
        }
    }

    pub fn after_op(ctx: &Context, op: Ptr<Operation>) -> ProgramPoint {
        if let Some(block) = op.deref(ctx).get_parent_block() {
            if let Some(next) = op.deref(ctx).get_next() {
                ProgramPoint::BeforeOpInBlock(block, next)
            } else {
                ProgramPoint::EndOfBlock(block)
            }
        } else {
            ProgramPoint::Operation(op)
        }
    }

    pub fn at_block_end(_ctx: &Context, block: Ptr<BasicBlock>) -> ProgramPoint {
        ProgramPoint::EndOfBlock(block)
    }

    pub fn next_op(&self, _ctx: &Context) -> Option<Ptr<Operation>> {
        match self {
            ProgramPoint::Operation(op) => Some(*op),
            ProgramPoint::BeforeOpInBlock(_, op) => Some(*op),
            ProgramPoint::EndOfBlock(_) => None,
        }
    }

    pub fn prev_op(&self, ctx: &Context) -> Option<Ptr<Operation>> {
        match self {
            ProgramPoint::Operation(op) => Some(*op),
            ProgramPoint::BeforeOpInBlock(_, op) => op.deref(ctx).get_prev(),
            ProgramPoint::EndOfBlock(block) => block.deref(ctx).get_tail(),
        }
    }

    pub fn is_block_start(&self, ctx: &Context) -> bool {
        match self {
            ProgramPoint::Operation(_) => false,
            // true if no preceding op
            ProgramPoint::BeforeOpInBlock(_, op) => op.deref(ctx).get_prev().is_none(),
            // true if Empty block
            ProgramPoint::EndOfBlock(ptr) => ptr.deref(ctx).get_head().is_none(),
        }
    }

    pub fn is_block_end(&self, _ctx: &Context) -> bool {
        match self {
            ProgramPoint::Operation(_) | ProgramPoint::BeforeOpInBlock(..) => false,
            ProgramPoint::EndOfBlock(_) => true,
        }
    }

    pub fn block(&self) -> Option<Ptr<BasicBlock>> {
        match self {
            ProgramPoint::Operation(_) => None,
            ProgramPoint::BeforeOpInBlock(block, _) => Some(*block),
            ProgramPoint::EndOfBlock(block) => Some(*block),
        }
    }
}

pub trait AnalysisState: PrintableState + Sized + 'static {
    type Anchor: Printable + Clone + Hash + 'static;

    fn create(anchor: Self::Anchor) -> Self;
    fn add_dependency<A: 'static>(&self, point: ProgramPoint);
    fn on_update(&self, ctx: &Context, solver: &DataflowSolver);
}

pub trait DataflowAnalysis: Downcast {
    /// Verify analysis can be run on the solver. Should be used to verify required analyses are
    /// loaded.
    fn verify(&self, solver: &DataflowSolver, ctx: &Context, root: Ptr<Operation>) -> Result<()> {
        let _ = (solver, ctx, root);
        Ok(())
    }

    #[allow(clippy::result_unit_err)]
    fn initialize(
        &mut self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()>;

    #[allow(clippy::result_unit_err)]
    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()>;

    fn initialize_equivalent_lattice_anchor(
        &self,
        solver: &mut DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) {
        let _ = (solver, ctx, root);
    }
}
impl_downcast!(DataflowAnalysis);

pub trait PrintableState: Printable + Downcast {}
impl<T: Printable + 'static> PrintableState for T {}
impl_downcast!(PrintableState);

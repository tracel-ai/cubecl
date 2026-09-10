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
use foldhash::fast::FixedState;
use pliron::{
    basic_block::BasicBlock,
    graph::HasLabel,
    linked_list::{ContainsLinkedList, LinkedList},
    printable::Printable,
    verify_err_noloc,
};

use smallvec::SmallVec;
pub use solver::*;

pub mod control_flow_uniformity;
pub mod dead_code;
pub mod sccp;
pub mod sparse;
pub mod value_uniformity;

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
    use core::cell::{Ref, RefMut};

    use super::*;

    pub type SolverWorkItem = (ProgramPoint, TypeId);

    /// Read-only ref, can read but not update state.
    pub struct ReadRef<'a, T: PrintableState> {
        value: Rc<RefCell<dyn PrintableState>>,
        _ty: PhantomData<&'a T>,
    }

    impl<T: PrintableState> ReadRef<'_, T> {
        #[track_caller]
        pub fn deref(&self) -> Ref<'_, T> {
            Ref::map(self.value.borrow(), |value| {
                value.downcast_ref::<T>().unwrap()
            })
        }
    }

    /// Write-only ref, required for soundness in cases where multiple aliasing lattice elements are
    /// referenced at the same time. Mutation is only allowed through `update_state`.
    pub struct WriteRef<'a, T: PrintableState> {
        value: Rc<RefCell<dyn PrintableState>>,
        _ty: PhantomData<&'a T>,
    }

    impl<T: PrintableState> WriteRef<'_, T> {
        #[track_caller]
        fn deref(&self) -> RefMut<'_, T> {
            RefMut::map(self.value.borrow_mut(), |value| {
                value.downcast_mut::<T>().unwrap()
            })
        }
    }

    impl<T: PrintableState> PartialEq<WriteRef<'_, T>> for ReadRef<'_, T> {
        fn eq(&self, other: &WriteRef<'_, T>) -> bool {
            Rc::ptr_eq(&self.value, &other.value)
        }
    }

    impl<T: PrintableState> PartialEq<ReadRef<'_, T>> for WriteRef<'_, T> {
        fn eq(&self, other: &ReadRef<'_, T>) -> bool {
            Rc::ptr_eq(&self.value, &other.value)
        }
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

    type AnalysisStates = HashMap<u64, States>;
    type States = HashMap<TypeId, StateEntry>;
    type StateEntry = Rc<RefCell<dyn PrintableState>>;

    pub struct DataflowSolver {
        child_analyses: HashMap<TypeId, Box<dyn DataflowAnalysis>>,
        worklist: RefCell<VecDeque<SolverWorkItem>>,
        anchor_hash: FixedState,
        analysis_states: RefCell<AnalysisStates>,
        config: SolverConfig,
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

        #[track_caller]
        pub fn update_state<A: AnalysisState>(
            &self,
            ctx: &Context,
            state: &WriteRef<A>,
            update: impl FnOnce(&mut A) -> ChangeResult,
        ) {
            let mut state = state.deref();
            let changed = update(&mut state);
            if changed == ChangeResult::Changed {
                state.on_update(ctx, self);
            }
        }

        pub fn get_or_create<T: AnalysisState>(&self, anchor: T::Anchor) -> ReadRef<'_, T> {
            let anchor_hash = self.hash_anchor(&anchor);
            let mut states = self.analysis_states.borrow_mut();
            let state = states.entry(anchor_hash).or_default();
            let id = TypeId::of::<T>();
            let state = state
                .entry(id)
                .or_insert_with(|| Rc::new(RefCell::new(T::create(anchor))))
                .clone();
            ReadRef {
                value: state,
                _ty: PhantomData,
            }
        }

        pub fn get_or_create_mut<T: AnalysisState>(&self, anchor: T::Anchor) -> WriteRef<'_, T> {
            let anchor_hash = self.hash_anchor(&anchor);
            let mut states = self.analysis_states.borrow_mut();
            let state = states.entry(anchor_hash).or_default();
            let id = TypeId::of::<T>();
            let state = state
                .entry(id)
                .or_insert_with(|| Rc::new(RefCell::new(T::create(anchor))))
                .clone();
            WriteRef {
                value: state,
                _ty: PhantomData,
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

        pub fn lookup_state<T: AnalysisState>(&self, anchor: T::Anchor) -> Option<ReadRef<'_, T>> {
            let anchor_hash = self.hash_anchor(&anchor);
            let states = self.analysis_states.borrow();
            let state = states.get(&anchor_hash)?.get(&TypeId::of::<T>())?.clone();
            Some(ReadRef {
                value: state,
                _ty: PhantomData,
            })
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
            let states = self.analysis_states.borrow();
            let mut entries = states
                .values()
                .flat_map(|states| states.iter())
                .collect::<Vec<_>>();
            entries.sort_by_key(|it| it.0);

            for (_, state) in entries {
                writeln!(f, "    {},", state.borrow().disp(ctx))?;
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

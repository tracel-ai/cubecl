//! What a walk's slots hold ([`Payload`]).

use cubecl::prelude::*;

use crate::*;

/// One operand as a slot plans it.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageOperand {
    /// Who moves this operand's bytes into its stage.
    pub(crate) delivery: Delivery,
    /// The dimensions the operand spans.
    pub(crate) space: Space,
    /// The keys of a table the operand has yet to read ([`Indexing`]): its window moves along
    /// them too, though it spans none of them.
    pub(crate) keys: Vec<Dimension>,
}

/// What shapes a slot's buffers: level, depth, storage, optional line width, and who holds them.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct StageSpec {
    pub(crate) level: Level,
    pub(crate) depth: usize,
    pub(crate) storage: StageStorage,
    pub(crate) width: Option<usize>,
    /// The element a stage that decodes its operand holds; `f32` where `None`.
    pub(crate) decoded: Option<ElemType>,
    pub(crate) owner: StageOwner,
}

/// One operand or two, staged and filled as one.
#[cube]
pub(crate) trait Payload<Me: CubeType>: CubeType {
    /// These operands as a slot plans them, in the order this payload holds them: `None` for one
    /// read where it lies.
    fn operands(&self) -> comptime_type!(Vec<Option<StageOperand>>);

    /// A staged copy of these operands, reusing `first`'s buffer where `refills` marks it
    /// [`Shared`](Refill::Shared).
    fn staged(
        &self,
        first: &Me,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> Me;

    /// Free the shared memory this payload's stages hold, for what is declared after it to take.
    fn free(&self);

    /// This payload as its reader takes it at `region`: what it stages, and `src`'s window of each
    /// operand `refills` reads in place ([`InPlace`](Refill::InPlace)).
    #[allow(dead_code)] // Reached through its expand, from the schedules.
    fn placed_at(&self, src: &Me, region: &Region, #[comptime] refills: Vec<Refill>) -> Me;

    /// These operands read where a walk proved its region inside their runs
    /// ([`Tile::inside_run`]).
    #[allow(dead_code)] // Reached through its expand, from the schedules' steady laps.
    fn inside_runs(&self) -> Me;

    /// Bring `src`'s window at `region` into this payload, for each operand whose refill is `only`.
    fn bring(
        &mut self,
        src: &Me,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    );
}

/// One operand's stage, placed in the operand's nest at the depth the walk's regions sit below,
/// `contracted` the Dimension a contraction sums it along where it is one's factor. The nest's
/// levels come with it: a plane's window of the stage is the plane's, and a landing out of it
/// takes a window for every plane the levels lay out.
#[cube]
pub(crate) fn stage_one(
    operand: &Tile,
    #[comptime] spec: StageSpec,
    #[comptime] contracted: Option<Dimension>,
) -> Tile {
    // An operand whose table this level reads stages one region of it: one slab, not the stack.
    let shaped = operand.shaped_for(spec.level.clone());
    let operand = &shaped;
    let define!(Staged) = operand.staged_element(spec.decoded);
    let stage = Memory::stage::<Staged>(
        operand,
        &spec.level,
        spec.storage.clone(),
        spec.width,
        spec.owner,
        spec.decoded,
        contracted,
    );
    let stage = Tile::new::<Staged>(
        stage.kind,
        comptime!(Placement::new(
            stage.place.space.clone(),
            spec.depth,
            operand.place.levels.clone()
        )),
    );
    stage.with_scales_staged(operand, comptime!(spec))
}

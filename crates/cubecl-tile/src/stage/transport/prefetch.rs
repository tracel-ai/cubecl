//! The straight transport split around a contraction: lines fetched into registers before it,
//! stored into the stage after.

use cubecl::prelude::*;

use super::padded::read_stage_line;
use super::straight::fill_extent;
use crate::*;

/// Most scalars of both operands' stages one unit may hold in registers beside an accumulator.
pub(crate) const MOST_FETCHED_SCALARS: usize = 128;

/// The lines of one stage a single unit moves: unit `u` takes lines `u`, `u + units`, ….
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct UnitLines {
    /// The units the lines are spread over: the launch's `CUBE_DIM`, or one plane's width.
    fill: FillUnits,
    /// The stage's line count.
    lines: usize,
}

impl UnitLines {
    /// The lines of a stage of `lines` spread over `units`; panics if `units` is zero.
    pub(crate) fn new(lines: usize, units: usize) -> Self {
        UnitLines::of(lines, FillUnits::cube(units))
    }

    /// [`new`](Self::new) for the units `fill` names.
    pub(crate) fn of(lines: usize, fill: FillUnits) -> Self {
        assert!(
            fill.count > 0,
            "UnitLines: a stage fetched into registers spreads its lines over the launch's \
             units, which this operand's spec does not state: bind it through `Launcher::arg`, \
             or set its `units`"
        );
        UnitLines { fill, lines }
    }

    /// Lines one unit moves.
    pub(crate) fn tasks(self) -> usize {
        self.lines.div_ceil(self.fill.count)
    }

    /// Scalars one unit holds in registers when each line is `width` wide.
    pub(crate) fn scalars(self, width: usize) -> usize {
        self.tasks() * width
    }

    /// Whether task `t` can run past the stage.
    fn guarded(self, t: usize) -> bool {
        (t + 1) * self.fill.count > self.lines
    }
}

#[cube]
impl Memory {
    /// The lines of this stage one unit moves; the stage shape must fold.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn unit_lines(&self) -> comptime_type!(UnitLines) {
        let folded = self.stage_lines().constant();
        let fill = comptime!(self.access.fill);
        comptime!(UnitLines::of(
            folded.expect("Memory: a stage fetched into registers has a static shape") as usize,
            fill,
        ))
    }

    /// Scalars of this stage one unit holds in registers between fetch and store.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(usize) {
        let lines = self.unit_lines();
        comptime!(lines.scalars(self.store.vector_size))
    }

    /// This stage's line count.
    fn stage_lines(&self) -> usize {
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    /// Read this unit's lines of `src` into `fetched`; only a plain, direct, whole stage at its
    /// source's width.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_straight<T: Numeric, W: Size>(
        &self,
        src: &Memory,
        #[comptime] space: Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        self.refuse_unfetchable(src);
        let w = comptime!(self.store.vector_size);
        let fetched_width = W::value();
        comptime!(assert_eq!(
            fetched_width, w,
            "Memory::fetch_straight: registers of one width fetch a stage of another"
        ));
        let check = comptime!(src.access.overhang.masks());
        comptime!(fill_extent(&space, w, w, check));
        let layout = self.layout.clone();
        let lines = self.unit_lines();
        let total = self.stage_lines();
        let s = Masked::new(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        #[unroll]
        for t in 0..comptime!(lines.tasks()) {
            let i = task_line(t, comptime!(lines));
            if in_stage(i, total, t, comptime!(lines)) {
                fetched[t] =
                    read_stage_line::<T, W, W>(&s, &layout.line_coords(i), comptime!(None));
            }
        }
    }

    /// Write this unit's `fetched` lines into the stage. The caller must `sync_cube` before (after
    /// the stage's last read) and after (before the next read).
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_fetched<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let lines = self.unit_lines();
        let total = self.stage_lines();
        let layout = self.layout.clone();
        let d = self.lines_storage_mut::<T, W>();
        #[unroll]
        for t in 0..comptime!(lines.tasks()) {
            let i = task_line(t, comptime!(lines));
            if in_stage(i, total, t, comptime!(lines)) {
                d[layout.line_offset(i)] = fetched[t];
            }
        }
    }

    /// Refuse a pairing this transport cannot move.
    fn refuse_unfetchable(&self, src: &Memory) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && src.store.packing == Packing::Plain
                && self.access.whole
                && !self.access.overhang.masks()
                && self.access.write == Write::Replace
                && self.store.vector_size == src.store.vector_size
                && self.projection.is_direct()
                && src.projection.is_direct(),
            "Memory::fetch_straight: only a plain, direct, whole stage filled at its source's \
             width is fetched into registers"
        ));
    }
}

/// The line task `t` of this unit moves.
#[cube]
fn task_line(#[comptime] t: usize, #[comptime] lines: UnitLines) -> usize {
    FillUnits::worker(comptime!(lines.fill)) + comptime!(t * lines.fill.count)
}

/// Whether line `i` is inside the stage; a constant where task `t` cannot run past it.
#[cube]
fn in_stage(i: usize, total: usize, #[comptime] t: usize, #[comptime] lines: UnitLines) -> bool {
    if comptime!(lines.guarded(t)) {
        i < total
    } else {
        true.runtime()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_unit_runs_the_same_tasks() {
        assert_eq!(UnitLines::new(256, 64).tasks(), 4);
        assert_eq!(UnitLines::new(255, 64).tasks(), 4);
        assert_eq!(UnitLines::new(1, 64).tasks(), 1);
    }

    #[test]
    fn the_registers_are_the_tasks_at_the_width() {
        assert_eq!(UnitLines::new(256, 64).scalars(4), 16);
        assert_eq!(UnitLines::new(256, 64).scalars(1), 4);
    }

    #[test]
    fn only_a_last_task_past_the_lines_is_guarded() {
        let exact = UnitLines::new(256, 64);
        for t in 0..exact.tasks() {
            assert!(
                !exact.guarded(t),
                "task {t} of an exact spread needs no guard"
            );
        }
        let ragged = UnitLines::new(200, 64);
        assert_eq!(ragged.tasks(), 4);
        for t in 0..3 {
            assert!(!ragged.guarded(t), "task {t} lies wholly inside the lines");
        }
        assert!(ragged.guarded(3), "the last task runs past 200 lines");
    }

    #[test]
    #[should_panic(expected = "spreads its lines over the launch's units")]
    fn lines_with_no_units_are_refused() {
        UnitLines::new(256, 0);
    }
}

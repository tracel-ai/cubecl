//! The tensor-core encoding of a plane tile ([`CmmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout, MmaDefinition},
    ir::{ElemType, FloatKind, FragmentPositions, Scope},
    prelude::*,
    unexpanded,
};

use super::mma::widths::NA;
use crate::ops::matmul::leaf::cmma::laid_fragment_read;
use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct CmmaData {
    /// The fragment, kept at [`Held`] and named at the element it holds where it is used.
    pub fragment: Matrix<Held>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    /// The whole MMA tile's `(m, n)`, whatever the role.
    #[cube(comptime)]
    pub shape: (usize, usize),
    /// This plane's one-tile window of shared memory the fragment bounces through, at the
    /// fragment's element.
    pub scratch: ComptimeOption<Raw<SharedSlice>>,
}

#[cube]
impl CmmaData {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, whatever the role.
    pub(crate) fn alloc<T: Numeric>(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> CmmaData {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        CmmaData {
            fragment: unsafe { matrix.downcast_unchecked::<Held>() },
            ident,
            layout,
            shape: comptime!((m, n)),
            scratch: ComptimeOption::new_None(),
        }
    }

    /// This fragment carrying `scratch`, one tile of shared memory for its plane.
    pub(crate) fn with_scratch<T: Numeric>(self, scratch: Shared<[T]>) -> CmmaData {
        CmmaData {
            fragment: self.fragment,
            ident: comptime!(self.ident),
            layout: comptime!(self.layout),
            shape: comptime!(self.shape),
            scratch: ComptimeOption::new_Some(Raw::<SharedSlice>::of::<T>(scratch)),
        }
    }

    /// Store this tile row-major into `scratch`, one tile's cells from its start.
    pub(crate) fn store_scratch<T: Numeric>(&self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        let mut window = scratch.clone();
        cmma::store(
            &mut window,
            &self.fragment.downcast::<T>(),
            n,
            MatrixLayout::RowMajor,
        )
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch<T: Numeric>(&mut self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        cmma::load_with_layout(
            &mut self.fragment.downcast::<T>(),
            scratch,
            n,
            MatrixLayout::RowMajor,
        )
    }

    /// Zero the fragment.
    pub(crate) fn zero<T: Numeric>(&mut self) {
        cmma::fill(&mut self.fragment.downcast::<T>(), T::from_int(0));
    }

    /// Fill this fragment from `mem`'s window, stepping rows by the physical row stride.
    pub(crate) fn load_window<T: Numeric>(&mut self, mem: &Memory, #[comptime] row: usize) {
        let element = mem.stage_element();
        comptime!(assert!(
            element.holds_values(),
            "CmmaData::load_window: a cmma fragment loads at one element type, so it cannot \
             unpack a packed source as it reads; contract it, which lands it, or decode it into a \
             stage (`stage.dequantize_from(&w.mul(&scales))`)"
        ));
        let stride = mem.row_stride_at(row);
        match comptime!(self.ident) {
            MatrixIdent::Accumulator => cmma::load_with_layout(
                &mut self.fragment.downcast::<T>(),
                mem.window_slice::<T>(),
                stride,
                comptime!(self.layout),
            ),
            _ => cmma::load(
                &mut self.fragment.downcast::<T>(),
                mem.window_slice::<T>(),
                stride,
            ),
        }
    }

    /// Fill this fragment from `mem`'s window over `space`: stepping rows along `row` in the
    /// layout it was opened with where the window lies in its Space's order, and as its stage lays
    /// it where the stage lies along a run of `k` ([`laid_fragment_read`]).
    pub(crate) fn load_window_over<T: Numeric>(
        &mut self,
        mem: &Memory,
        #[comptime] space: &Space,
        #[comptime] row: usize,
    ) {
        let laid = comptime!(laid_fragment_read(self.ident, space, mem.layout.order));
        match laid {
            Some((layout, row)) => self.load_laid_window::<T>(mem, row, layout),
            None => self.load_window::<T>(mem, row),
        }
    }

    /// Fill this fragment from `mem`'s window read as `layout`, its rows stepping along `row`.
    fn load_laid_window<T: Numeric>(
        &mut self,
        mem: &Memory,
        #[comptime] row: usize,
        #[comptime] layout: MatrixLayout,
    ) {
        let stride = mem.row_stride_at(row);
        cmma::load_with_layout(
            &mut self.fragment.downcast::<T>(),
            mem.window_slice::<T>(),
            stride,
            layout,
        )
    }

    /// Drain this fragment into `mem`'s *window* (origin offset, physical row stride).
    pub(crate) fn store_window<T: Numeric>(&self, mem: &mut Memory, #[comptime] row: usize) {
        let stride = mem.row_stride_at(row);
        cmma::store(
            mem.window_slice_mut::<T>(),
            &self.fragment.downcast::<T>(),
            stride,
            comptime!(self.layout),
        )
    }

    /// This fragment, whose first cell lies at `origin` of the sum's window, with each cell under
    /// its `scales`, bounced through the fragment's slot of the plane's scratch as
    /// [`mul_along`](Self::mul_along) bounces it: a cmma fragment is not read cell by cell.
    pub(crate) fn scale<T: Numeric>(
        &self,
        scales: &mut CellScales,
        #[comptime] origin: (usize, usize),
    ) {
        self.bounced::<T>(|| self.scale_spilled::<T>(scales, origin));
    }

    /// Each of this fragment's spilled cells under its `scales`, the fragment's first cell at
    /// `origin` of the sum's window, the plane's units taking them in turns. The caller owns the
    /// barriers.
    fn scale_spilled<T: Numeric>(
        &self,
        scales: &mut CellScales,
        #[comptime] origin: (usize, usize),
    ) {
        let mut scratch = self.scratch_slot::<T>();
        let (m, n) = self.shape;
        in_plane_turns(m * n, |nth| {
            let mut cells = Sequence::<HeldCell>::new();
            cells.push(HeldCell {
                row: (nth / n) as u32 + comptime!(origin.0 as u32),
                col: (nth % n) as u32 + comptime!(origin.1 as u32),
                first_unit: comptime!(None),
            });
            let cell = Vector::<T, Const<1>>::cast_from(scratch[nth]);
            scratch[nth] = scales.applied::<T, Const<1>>(cell, &cells).extract(0usize);
        });
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the fragment's columns,
    /// bounced through its slot of the plane's scratch: spilled, scaled by the plane's units in
    /// turns, and reloaded, each step met on `sync_plane`. The slot is the plane's own, so no
    /// other plane waits; every unit of the plane calls it.
    pub(crate) fn mul_along<T: Numeric>(&self, factors: &Array<T>, #[comptime] first: usize) {
        self.bounced::<T>(|| self.mul_spilled(factors, first));
    }

    /// `slot[i, :] *= factors[first + i]` over this fragment's spilled cells, the plane's units
    /// taking them in turns. The caller owns the barriers.
    pub(crate) fn mul_spilled<T: Numeric>(&self, factors: &Array<T>, #[comptime] first: usize) {
        let mut scratch = self.scratch_slot::<T>();
        let (m, n) = self.shape;
        in_plane_turns(m * n, |cell| {
            scratch[cell] *= factors[first + cell / n];
        });
    }

    /// Load this fragment back from its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn reload_from_scratch<T: Numeric>(&self) {
        let scratch = self.scratch_slot::<T>();
        let mut fragment = self.clone();
        fragment.load_scratch(&scratch);
    }

    /// Store this fragment into its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn spill_to_scratch<T: Numeric>(&self) {
        self.store_scratch(&self.scratch_slot::<T>());
    }

    /// Write this fragment's spilled cells into `mem` through the store's own write
    /// ([`write_cells`]), the plane's units taking its runs in turns: each run read out of the
    /// scratch in one load and written a store's line at a time ([`ScratchRuns`]). Each cell is
    /// added once into a destination that adds, and written once into any other.
    ///
    /// # Assumptions
    ///
    /// - The scratch slot holds the whole fragment, row-major, and starts on 16 bytes.
    ///
    /// # Panics
    ///
    /// - Where the store's lines do not divide the fragment's columns.
    pub(crate) fn add_from_scratch<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: &Space,
    ) {
        let scratch = self.scratch_slot::<T>();
        let (m, n) = comptime!(self.shape);
        let width = comptime!(mem.store.vector_size);
        let runs = comptime!(ScratchRuns::new(m, n, width));
        let size!(W) = width;
        let size!(R) = runs.run;
        let lines = runs.lines_per_run();
        let read = scratch.as_vectorized().with_vector_size::<R>();
        write_cells::<Out, W>(mem, space, |sink| {
            in_plane_turns(comptime!(runs.count()), |run| {
                let cells = read[run];
                let (row, first_line) = scratch_run_place(comptime!(runs), run);
                #[unroll]
                for l in 0..lines {
                    let mut value = Vector::<Out, W>::empty();
                    #[unroll]
                    for e in 0..width {
                        value.insert(e, Out::cast_from(cells.extract(l * width + e)));
                    }
                    sink.write((row, first_line + l as u32), value);
                }
            });
        });
    }

    /// Whether this fragment's sum reaches a destination bound `width` values a line written out
    /// of its units' registers where it would otherwise bounce ([`FragmentDrain::Cells`]): where
    /// the target states which unit holds which cell, and its units, alone or regrouped, hold
    /// lines `width` wide along the destination's rows ([`CellLines`]).
    pub(crate) fn drains_by_cell<T: Numeric>(
        &self,
        #[comptime] width: usize,
    ) -> comptime_type!(bool) {
        let target = comptime::target_properties().comptime();
        let (m, n) = comptime!(self.shape);
        // The depth names no cell of an accumulator.
        let def = MmaDefinition::<T, T, T>::new(m, n, m);
        let elems = def.elems_per_lane(MatrixIdent::Accumulator);
        let positions = def.element_positions(MatrixIdent::Accumulator);
        let layout = comptime!(self.layout);
        let lanes = comptime!(target.mma.const_plane_lane_count as usize);
        comptime!(
            target.mma.fragment_positions == FragmentPositions::AsElementPositions
                && positions.is_some_and(|positions| {
                    CellLines::of(&positions, lanes, elems, layout, width).is_some()
                })
        )
    }

    /// Write this fragment's cells into `mem` through the store's own write ([`write_cells`]),
    /// each unit the ones its registers hold ([`Matrix::to_registers`]), cast to the store's
    /// element: a fragment whose positions the target states drains so where it would otherwise
    /// bounce through a scratch.
    ///
    /// # Assumptions
    ///
    /// - The target states which unit holds which cell of a fragment
    ///   ([`drains_by_cell`](Self::drains_by_cell)).
    pub(crate) fn store_cells<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: &Space,
    ) {
        let (m, n) = comptime!(self.shape);
        // The depth names no cell of an accumulator.
        let def = MmaDefinition::<T, T, T>::new(m, n, m);
        register_acc_size::<T>(&def);
        let registers = self.fragment.downcast::<T>().to_registers::<NA>();
        store_cells::<T, Out, T, T, T>(mem, &registers, &def, comptime!(self.layout), space);
    }

    /// Drain this fragment through its slot of the plane's scratch. The slot is the plane's own,
    /// so the plane alone meets around it, as [`scale`](Self::scale) does; ordering `mem` against
    /// other planes is its caller's, as for every other drain.
    pub(crate) fn bounce_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: &Space,
    ) {
        sync_plane();
        self.spill_to_scratch::<T>();
        sync_plane();
        self.add_from_scratch::<T, Out>(mem, space);
        sync_plane();
    }

    /// This fragment's slot of the plane's scratch, or the reason there is none.
    fn scratch_slot<T: Numeric>(&self) -> Shared<[T]> {
        #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.at::<T>().clone(),
            ComptimeOption::None => panic!(
                "CmmaData: a fragment bounces through a scratch, to scale its slices, to reach a \
                 store that sums, or a window the problem's edge cuts short; open the \
                 accumulator with `with_scratch`"
            ),
        }
    }

    /// Drain this fragment into `mem`'s window, casting `T` down to the sink's element type.
    pub(crate) fn store_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] row: usize,
    ) {
        let stride = mem.row_stride_at(row);
        let casted: Matrix<Out> = cmma::cast(&self.fragment.downcast::<T>());
        cmma::store(
            mem.window_slice_mut::<Out>(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}

impl CmmaData {
    /// Runs `work` on this fragment's cells spilled into its slot of the plane's scratch, then
    /// reloads it, each step met on `sync_plane`: the slot is the plane's own, so no other plane
    /// waits, and every unit of the plane calls it.
    // `T` is read by the expansion, which spills and reloads the fragment at it.
    #[allow(clippy::extra_unused_type_parameters)]
    pub(crate) fn bounced<T: Numeric>(&self, _work: impl FnOnce()) {
        unexpanded!()
    }
}

impl CmmaDataExpand {
    pub(crate) fn __expand_bounced_method<T: Numeric>(
        &self,
        scope: &Scope,
        work: impl FnOnce(&Scope),
    ) {
        self.__expand_spill_to_scratch_method::<T>(scope);
        sync_plane::expand(scope);
        work(scope);
        sync_plane::expand(scope);
        self.__expand_reload_from_scratch_method::<T>(scope);
        sync_plane::expand(scope);
    }
}

/// Whether a fragment of `T` cast to `Out` is a fragment the device holds at `(m, n)`: the same
/// element, or one an accumulator is offered in at that shape. `wmma` offers none in `bf16`, so a
/// fragment cast there is a kernel that fails to compile, and a `bf16` sink takes the cells through
/// the scratch instead, each cast on its write. A device whose properties are not known at
/// expansion is taken at its word.
// `T` and `Out` are read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
pub(crate) fn casts_in_place<T: Numeric, Out: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
) -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let (from, to) = (T::elem_type(scope), Out::elem_type(scope));
        from == to
            || scope
                .state()
                .device_properties
                .as_ref()
                .is_none_or(|properties| {
                    cmma_cast_layout_is_compatible(&properties.identity.fingerprint, from, to)
                        && properties.features.matmul.cmma.iter().any(|offered| {
                            offered.cd_type == to
                                && offered.m as usize == m
                                && offered.n as usize == n
                        })
                })
    })
}

/// Volta uses different register layouts for f32 and f16 accumulator fragments, so cross-type
/// casts must bounce through shared memory.
fn cmma_cast_layout_is_compatible(fingerprint: &str, from: ElemType, to: ElemType) -> bool {
    let f32 = ElemType::Float(FloatKind::F32);
    let f16 = ElemType::Float(FloatKind::F16);
    !(fingerprint == "ptx_sm70" && ((from == f32 && to == f16) || (from == f16 && to == f32)))
}

#[cfg(test)]
mod accumulator_cast_layout_tests {
    use super::*;

    #[test]
    fn volta_bounces_f32_f16_accumulator_casts() {
        let (f32, f16) = (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F16),
        );
        assert!(!cmma_cast_layout_is_compatible("ptx_sm70", f32, f16));
        assert!(!cmma_cast_layout_is_compatible("ptx_sm70", f16, f32));
    }

    #[test]
    fn later_architectures_keep_the_in_place_cast_path() {
        let (f32, f16) = (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F16),
        );
        assert!(cmma_cast_layout_is_compatible("ptx_sm75", f32, f16));
        assert!(cmma_cast_layout_is_compatible("ptx_sm80", f32, f16));
    }

    #[test]
    fn same_type_cast_does_not_depend_on_architecture() {
        let f16 = ElemType::Float(FloatKind::F16);
        assert!(cmma_cast_layout_is_compatible("ptx_sm70", f16, f16));
    }
}

/// How a cmma fragment reaches its destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FragmentDrain {
    /// The fragment's store intrinsic: a replacing destination, window wholly inside it.
    Intrinsic,
    /// Each unit's cells out of its registers, where the target states their positions: a
    /// destination the intrinsic cannot store to, one value a line or a unit's run a line.
    Cells,
    /// Through the plane's scratch: a destination that adds, or a window cut short by the edge.
    Bounce,
}

impl FragmentDrain {
    /// How a fragment drains into a destination written as `access` says, `addressed` when its
    /// values sit at an address the intrinsic can store to, `in_place` when the fragment cast to
    /// the destination's element is one the device holds ([`casts_in_place`]), and `by_cell` when
    /// its units can write its cells out of their registers ([`CmmaData::drains_by_cell`]).
    pub(crate) const fn of(
        access: &Access,
        addressed: bool,
        in_place: bool,
        by_cell: bool,
    ) -> Self {
        match (
            access.write,
            access.overhang,
            addressed && in_place,
            by_cell,
        ) {
            (Write::Replace, Overhang::Never | Overhang::Fits, true, _) => FragmentDrain::Intrinsic,
            (_, _, _, true) => FragmentDrain::Cells,
            (Write::Replace, _, _, false)
            | (Write::Accumulate | Write::Exclusive(_), _, _, false) => FragmentDrain::Bounce,
        }
    }
}

/// How a fragment's spilled cells are read out of its scratch: in runs of `run` cells along a
/// row, each read in one load and written as whole lines of the store, a run never splitting a
/// line.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct ScratchRuns {
    /// Cells one run reads.
    pub(crate) run: usize,
    /// The fragment's rows.
    pub(crate) rows: usize,
    /// The fragment's columns.
    pub(crate) cols: usize,
    /// Cells one line of the store holds.
    pub(crate) width: usize,
}

impl ScratchRuns {
    /// The cells a 16-byte load reads of a 32-bit sum, at least.
    const LEAST_RUN_CELLS: usize = 4;

    /// The runs of a `rows × cols` fragment written into a store of `width` cells a line: runs of
    /// the widest of a line and four cells, as wide as a row at most.
    ///
    /// # Panics
    ///
    /// - Where the store's lines do not divide the fragment's columns.
    pub(crate) fn new(rows: usize, cols: usize, width: usize) -> Self {
        assert!(
            cols.is_multiple_of(width),
            "CmmaData::add_from_scratch: the store's lines ({width}) should divide the \
             fragment's columns ({cols})"
        );
        let run = width.max(Self::LEAST_RUN_CELLS).min(cols);
        Self {
            run,
            rows,
            cols,
            width,
        }
    }

    /// The runs the fragment's cells are read in.
    pub(crate) fn count(&self) -> usize {
        self.rows * self.cols / self.run
    }

    /// The store's lines one run holds.
    pub(crate) fn lines_per_run(&self) -> usize {
        self.run / self.width
    }

    /// The runs one row holds.
    pub(crate) fn runs_per_row(&self) -> usize {
        self.cols / self.run
    }
}

/// The row of `runs`' run `run`, and the first of the store's lines along it the run covers.
#[cube]
fn scratch_run_place(#[comptime] runs: ScratchRuns, run: usize) -> (u32, u32) {
    let runs_per_row = comptime!(runs.runs_per_row());
    let lines = comptime!(runs.lines_per_run());
    (
        (run / runs_per_row) as u32,
        (run % runs_per_row * lines) as u32,
    )
}

#[cfg(test)]
mod fragment_drain_tests {
    use super::*;

    fn access(write: Write, overhang: Overhang) -> Access {
        Access {
            whole: false,
            overhang,
            write,
            copy: Level::copied_by_cube(0),
            storage: WindowStorage::Stored(Storage::Strided),
            delivery: Delivery::SyncPerUnit,
        }
    }

    #[test]
    fn a_replacing_window_inside_its_buffer_stores_through_the_intrinsic() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            for by_cell in [false, true] {
                let access = access(Write::Replace, overhang);
                let drain = FragmentDrain::of(&access, true, true, by_cell);
                assert_eq!(drain, FragmentDrain::Intrinsic);
            }
        }
    }

    #[test]
    fn an_overhanging_window_bounces() {
        let masked =
            FragmentDrain::of(&access(Write::Replace, Overhang::Masked), true, true, false);
        assert_eq!(masked, FragmentDrain::Bounce);
    }

    #[test]
    fn a_summing_window_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits, Overhang::Masked] {
            for write in [Write::Accumulate, Write::Exclusive(Schedule::Sequential)] {
                assert_eq!(
                    FragmentDrain::of(&access(write, overhang), true, true, false),
                    FragmentDrain::Bounce,
                    "{write:?} {overhang:?}"
                );
            }
        }
    }

    /// Where the target states which unit holds which cell, a destination the intrinsic cannot
    /// store to takes each unit's cells out of its registers rather than through a scratch.
    #[test]
    fn cmma_drains_by_cell_where_positions_are_stated() {
        let summing = access(Write::Accumulate, Overhang::Never);
        let masked = access(Write::Replace, Overhang::Masked);
        for access in [summing, masked] {
            let drain = FragmentDrain::of(&access, true, true, true);
            assert_eq!(drain, FragmentDrain::Cells, "{access:?}");
        }
    }

    /// A fragment cast to an element the device holds no fragment of cannot be stored whole, so
    /// even a replacing window wholly inside its buffer bounces, its cells cast on their write.
    #[test]
    fn a_fragment_with_no_cast_in_place_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), true, false, false);
            assert_eq!(drain, FragmentDrain::Bounce);
        }
    }

    /// A write call has no address for the intrinsic to store to, so even a replacing window
    /// wholly inside it bounces.
    #[test]
    fn a_window_with_no_address_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), false, true, false);
            assert_eq!(drain, FragmentDrain::Bounce);
        }
    }
}

#[cfg(test)]
mod scratch_runs_tests {
    use super::*;

    /// Every cell of the fragment is in exactly one run, which a plane's units take in turns
    /// whatever its width: each run once, its lines the store's.
    #[test]
    fn add_from_scratch_runs_match_cells() {
        for (cols, width) in [(8, 1), (16, 1), (16, 2), (16, 8), (8, 8)] {
            for plane in [32usize, 64] {
                let runs = ScratchRuns::new(16, cols, width);
                let mut seen = vec![0usize; 16 * cols];
                for unit in 0..plane {
                    for run in (unit..runs.count()).step_by(plane) {
                        let row = run / runs.runs_per_row();
                        let line = run % runs.runs_per_row() * runs.lines_per_run();
                        for cell in 0..runs.run {
                            let col = line * width + cell;
                            assert_eq!(run * runs.run + cell, row * cols + col, "{cols} {width}");
                            seen[row * cols + col] += 1;
                        }
                    }
                }
                assert!(seen.iter().all(|&n| n == 1), "{cols} {width} {plane}");
            }
        }
    }

    /// A run reads at least a 16-byte load of a 32-bit sum, and never splits a store's line.
    #[test]
    fn a_run_is_a_whole_line_at_least_four_cells() {
        assert_eq!(ScratchRuns::new(16, 16, 1).run, 4);
        assert_eq!(ScratchRuns::new(16, 16, 8).run, 8);
        assert_eq!(ScratchRuns::new(16, 2, 1).run, 2);
    }
}

#[cfg(test)]
mod held_fragment_tests {
    use half::f16;

    use super::*;

    use crate::test_scope::test_scope;

    /// A `16 × 16` accumulator fragment allocated at `H`, kept at [`Held`], and stored into a
    /// scratch of `N`.
    #[cube]
    fn fragment_stored_at<H: Numeric, N: Numeric>() {
        let data = CmmaData::alloc::<H>(
            MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            MatrixLayout::Undefined,
        );
        let scratch = Shared::<[N]>::new_slice(256usize);
        data.store_scratch::<N>(&scratch);
    }

    /// A fragment kept at [`Held`] is named again at the element it was allocated at.
    #[test]
    fn a_held_fragment_is_named_at_its_element() {
        fragment_stored_at::expand::<f16, f16>(&test_scope());
    }

    /// A fragment kept at [`Held`] and named at an element it does not hold is refused at
    /// expansion.
    ///
    /// The tile keeps every fragment at one dummy type, so nothing in Rust ties a fragment to the
    /// element it was allocated at: only this check stops an `f16` fragment being stored as an
    /// `f32` one.
    #[test]
    #[should_panic(expected = "Matrix::downcast: the fragment holds")]
    fn a_held_fragment_refuses_another_element() {
        fragment_stored_at::expand::<f16, f32>(&test_scope());
    }
}

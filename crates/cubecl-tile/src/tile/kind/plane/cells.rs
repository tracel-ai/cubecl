//! How a plane's units reach the cells of a sum: in turns over a plane's cells ([`plane_turns`]),
//! written through a memory window's own write ([`cell_sink`]), and written out of the registers
//! that hold them as lines of their destination ([`CellLines`], [`store_cells`]).

use cubecl::cmma::{MatrixIdent, MatrixLayout, MmaDefinition};
use cubecl::ir::ElementPositions;
use cubecl::prelude::*;

use super::mma::widths::NA;
use crate::*;

/// This unit's first cell, and the cells between two of its turns, when a plane's units take a
/// sum's cells in turns, a cell each per turn: walked as `range_stepped(first, cells, stride)`.
///
/// # Guarantees
///
/// - Across a plane's units, that walk hands every cell below `cells` to exactly one unit, once,
///   for any plane width.
#[cube]
pub fn plane_turns() -> (usize, usize) {
    (UNIT_POS_PLANE as usize, PLANE_DIM as usize)
}

/// The sink a sum's cells are written into `mem` through, by its own write, which replaces them
/// or adds into them: each run of `W` cells at its row and line of the matrix `space`'s edges
/// make, its innermost dimension the columns and the last one above it longer than one cell the
/// rows, as the fragment's store intrinsic addresses them.
///
/// # Guarantees
///
/// - The sink addresses `mem`'s window as the accumulator's rows and columns, whatever order
///   `space` lists the window's dimensions in past its trailing ones, and whatever one-wide
///   dimensions it lists between them.
#[cube]
pub fn cell_sink<'a, Out: Numeric, W: Size>(
    mem: &'a mut Memory,
    #[comptime] space: &'a Space,
) -> MatrixViewMut<'a, Vector<Out, W>> {
    mem.matrix_mut::<Out, W>(0usize, comptime!(MatrixDimensions::edges(space)), space)
}

/// How a plane's units write the cells of a sum they hold as lines of `width` cells along the
/// rows of their destination: each unit `width` of its cells a line, as it holds them
/// (`holders == 1`), or, where each of `width` neighbouring units holds one cell of every line
/// of a block, after a regroup across them in registers that hands each unit whole lines of the
/// block (`holders == width`).
///
/// # Guarantees
///
/// - Where `holders == 1`, every unit holds its cells `width * b..width * (b + 1)` in order along
///   one row, the first at a column `width` divides.
/// - Where `holders == width`, the unit `p` of each group of `width` neighbouring units starting
///   at a unit `width` divides holds, at cell `e` of each block `width * b..width * (b + 1)`, the
///   cell `p` columns past the one the group's first unit holds there, on the same row; that
///   first column is one `width` divides.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellLines {
    /// Cells a line holds, a power of two.
    pub width: usize,
    /// The units that hold a line between them as the sum lies: `1` where a unit holds it whole,
    /// `width` where each holds one of its cells.
    pub holders: usize,
}

impl CellLines {
    /// How `lanes` units each holding `elems` cells where `positions` places them write them as
    /// lines `width` cells wide along the rows of a destination that lays the sum out as
    /// `layout`. `None` where neither a unit nor `width` neighbouring units together hold such a
    /// line, and for an [`Undefined`](MatrixLayout::Undefined) layout, which names no row.
    pub fn of(
        positions: &ElementPositions,
        lanes: usize,
        elems: usize,
        layout: MatrixLayout,
        width: usize,
    ) -> Option<Self> {
        let fits =
            width.is_power_of_two() && elems.is_multiple_of(width) && lanes.is_multiple_of(width);
        if !fits || layout == MatrixLayout::Undefined {
            return None;
        }
        let laid = |lane: usize, nth: usize| laid_position(positions.of_nth(lane, nth), layout);
        let blocks = || (0..elems).step_by(width);
        let held = (0..lanes).all(|lane| {
            blocks().all(|first| {
                let (row, col) = laid(lane, first);
                col.is_multiple_of(width)
                    && (0..width).all(|e| laid(lane, first + e) == (row, col + e))
            })
        });
        if held {
            return Some(Self { width, holders: 1 });
        }
        let shared = (0..lanes).step_by(width).all(|lane| {
            blocks().all(|first| {
                let col = laid(lane, first).1;
                col.is_multiple_of(width)
                    && (0..width).all(|e| {
                        let row = laid(lane, first + e).0;
                        (0..width).all(|p| laid(lane + p, first + e) == (row, col + p))
                    })
            })
        });
        shared.then_some(Self {
            width,
            holders: width,
        })
    }

    /// The widest line [`of`](Self::of) accepts for `lanes` units each holding `elems` cells
    /// where `positions` places them, along the rows of a destination laid out as `layout`: one
    /// cell where it accepts none wider.
    ///
    /// # Guarantees
    ///
    /// - `of` accepts every power of two up to the width returned, so a destination bound at any
    ///   width that divides it is written out of registers.
    pub fn widest(
        positions: &ElementPositions,
        lanes: usize,
        elems: usize,
        layout: MatrixLayout,
    ) -> usize {
        let accepted = |width: &usize| Self::of(positions, lanes, elems, layout, *width).is_some();
        core::iter::successors(Some(1usize), |width| width.checked_mul(2))
            .take_while(|width| *width <= elems.max(1))
            .take_while(accepted)
            .last()
            .unwrap_or(1)
    }
}

/// `(row, col)` of a sum as a destination laid out as `layout` holds it.
///
/// # Assumptions
///
/// - `layout` is row- or col-major.
fn laid_position((row, col): (usize, usize), layout: MatrixLayout) -> (usize, usize) {
    match layout {
        MatrixLayout::RowMajor => (row, col),
        MatrixLayout::ColMajor => (col, row),
        MatrixLayout::Undefined => unreachable!("CellLines: a sum should be row- or col-major"),
    }
}

/// The cell an instruction names at `(row, col)` where a sum laid out as `layout` holds it.
#[cube]
pub fn laid_out(row: u32, col: u32, #[comptime] layout: MatrixLayout) -> (u32, u32) {
    match layout {
        MatrixLayout::RowMajor => (row, col),
        MatrixLayout::ColMajor => (col, row),
        MatrixLayout::Undefined => panic!("mma: a sum is laid out row- or col-major"),
    }
}

/// Each unit's accumulator cells of `def`, held in `fragment` in the order
/// [`MmaDefinition::position_of_nth`] numbers them, written through `mem`'s own write
/// ([`cell_sink`]) a line of `mem`'s width at a time, so a destination that replaces, adds or
/// masks its cells takes each once. Where no unit holds a line, the units that hold it first
/// regroup it among them in registers ([`CellLines`]).
///
/// # Assumptions
///
/// - Every unit of the plane runs it, on the same path: the regroup reads every unit's cells.
///
/// # Guarantees
///
/// - Each cell is cast to `Out` and written once.
/// - It touches no shared memory and meets no barrier.
///
/// # Panics
///
/// - At expansion, where [`CellLines::of`] refuses `mem`'s width for the sum.
#[cube]
pub fn store_cells<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &mut Memory,
    fragment: &Array<Vector<T, NA>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] layout: MatrixLayout,
    #[comptime] space: &Space,
) {
    let width = comptime!(mem.store.vector_size);
    let lanes = comptime::target_properties()
        .comptime()
        .mma
        .const_plane_lane_count;
    let elems = def.elems_per_lane(MatrixIdent::Accumulator);
    let run = def.vector_size(MatrixIdent::Accumulator);
    let positions = def.element_positions(MatrixIdent::Accumulator);
    let lines = comptime!(
        positions
            .and_then(|positions| CellLines::of(&positions, lanes as usize, elems, layout, width))
            .unwrap_or_else(|| panic!(
                "store_cells: a sum should drain out of its units' registers only where its \
                 units, alone or {width} together, hold lines {width} cells wide along its rows"
            ))
    );
    let mut cells = Array::<T>::new(elems);
    #[unroll]
    for i in 0..comptime!(elems / run) {
        #[unroll]
        for e in 0..run {
            cells[i * run + e] = fragment[i].extract(e);
        }
    }
    regroup_cells::<T>(&mut cells, elems, lines.holders);
    let holders = comptime!(lines.holders as u32);
    let first_holder = UNIT_POS_PLANE / holders * holders;
    let place = UNIT_POS_PLANE % holders;
    let size!(W) = width;
    let mut sink = cell_sink::<Out, W>(mem, space);
    #[unroll]
    for line in 0..comptime!(elems / width) {
        // The line starts where the group's first unit holds the block's cell this unit took.
        let (row, col) = def.position_of_nth(
            first_holder,
            (line * width) as u32 + place,
            MatrixIdent::Accumulator,
        );
        let (row, col) = laid_out(row, col, layout);
        let mut value = Vector::<Out, W>::empty();
        #[unroll]
        for e in 0..width {
            value.insert(e, Out::cast_from(cells[line * width + e]));
        }
        sink.write((row, col / comptime!(width as u32)), value);
    }
}

/// Transposes each block of `holders` cells among each `holders` neighbouring units of the
/// plane, in registers: afterwards unit `p` of a group holds, at cell `e` of a block, what unit
/// `e` held at its cell `p`. A butterfly: each step swaps half a block's cells with the unit
/// whose place differs in one bit, through a plane shuffle, with no shared memory and no barrier.
#[cube]
fn regroup_cells<T: Numeric>(
    cells: &mut Array<T>,
    #[comptime] elems: usize,
    #[comptime] holders: usize,
) {
    #[unroll]
    for bit in 0..comptime!(holders.trailing_zeros() as usize) {
        let mask = 1usize << bit;
        let upper = (UNIT_POS_PLANE / comptime!(mask as u32)) % 2 == 1;
        #[unroll]
        for pair in 0..elems / 2 {
            let low = (pair / mask) * 2 * mask + pair % mask;
            let high = low + mask;
            let sent = select(upper, cells[low], cells[high]);
            let got = plane_shuffle_xor(sent, comptime!(mask as u32));
            cells[low] = select(upper, got, cells[low]);
            cells[high] = select(upper, cells[high], got);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{CoordinatePart, ElemType, ElementIndex, FloatKind};
    use cubecl_ir::amd::{AmdMatrixCore, AmdMfma, AmdWmma};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Where a matrix core's lanes hold a sum, how many lanes there are, and the cells each holds.
    type Held = (ElementPositions, usize, usize);

    /// Where `core`'s lanes hold a `16 × 16` `f32` sum.
    fn held_by(core: AmdMatrixCore) -> Held {
        let mma = core.mma_properties();
        let positions = mma
            .element_positions_of(MatrixIdent::Accumulator, (16, 16, 16).into(), F32)
            .expect("an AMD matrix core should state its positions");
        let lanes = mma.const_plane_lane_count as usize;
        (positions, lanes, 16 * 16 / lanes)
    }

    fn rdna3() -> Held {
        held_by(AmdMatrixCore::Wmma(AmdWmma::Rdna3))
    }

    fn rdna4() -> Held {
        held_by(AmdMatrixCore::Wmma(AmdWmma::Rdna4))
    }

    fn cdna3() -> Held {
        held_by(AmdMatrixCore::Mfma(AmdMfma::Cdna3))
    }

    /// Lanes each holding cells two columns apart, which no lane or group of lanes holds as a
    /// line wider than one cell.
    fn scattered() -> Held {
        let lane = CoordinatePart::of(ElementIndex::Lane);
        let nth = CoordinatePart::of(ElementIndex::Nth);
        let positions = ElementPositions {
            row: vec![lane.counted_over(16)],
            col: vec![nth.spaced_by(2), lane.in_runs_of(16)],
        };
        (positions, 32, 8)
    }

    fn of(
        (positions, lanes, elems): &Held,
        layout: MatrixLayout,
        width: usize,
    ) -> Option<CellLines> {
        CellLines::of(positions, *lanes, *elems, layout, width)
    }

    fn widest((positions, lanes, elems): &Held, layout: MatrixLayout) -> usize {
        CellLines::widest(positions, *lanes, *elems, layout)
    }

    /// An RDNA4 lane holds eight cells of a row, and writes lines that split it as it holds it.
    #[test]
    fn a_lane_holding_a_run_writes_it_as_it_lies() {
        for width in [1, 2, 4, 8] {
            assert_eq!(
                of(&rdna4(), MatrixLayout::RowMajor, width),
                Some(CellLines { width, holders: 1 })
            );
        }
        assert_eq!(widest(&rdna4(), MatrixLayout::RowMajor), 8);
    }

    /// An RDNA3 or CDNA lane holds one column of its rows, so a line is regrouped among the
    /// `width` lanes that hold its columns.
    #[test]
    fn lanes_holding_a_column_regroup_a_line_among_them() {
        let four = Some(CellLines {
            width: 4,
            holders: 4,
        });
        assert_eq!(of(&rdna3(), MatrixLayout::RowMajor, 4), four);
        assert_eq!(widest(&rdna3(), MatrixLayout::RowMajor), 8);
        assert_eq!(of(&cdna3(), MatrixLayout::RowMajor, 4), four);
        assert_eq!(widest(&cdna3(), MatrixLayout::RowMajor), 4);
    }

    /// A line wider than a lane's cells, or than the lanes that share its rows, is refused.
    #[test]
    fn a_line_no_lanes_hold_is_refused() {
        assert_eq!(of(&rdna3(), MatrixLayout::RowMajor, 16), None);
        assert_eq!(of(&rdna4(), MatrixLayout::RowMajor, 16), None);
        assert_eq!(of(&cdna3(), MatrixLayout::RowMajor, 8), None);
        assert_eq!(of(&scattered(), MatrixLayout::RowMajor, 2), None);
        assert_eq!(widest(&scattered(), MatrixLayout::RowMajor), 1);
    }

    /// Every power of two up to the widest line is accepted too, for both layouts of every
    /// lanes' positions, so a destination bound at any width dividing it drains from registers.
    #[test]
    fn widest_line_halves_stay_accepted() {
        for held in [rdna3(), rdna4(), cdna3(), scattered()] {
            for layout in [MatrixLayout::RowMajor, MatrixLayout::ColMajor] {
                let widest = widest(&held, layout);
                for width in (0..=widest.ilog2()).map(|bits| 1usize << bits) {
                    assert!(of(&held, layout, width).is_some(), "{layout:?} {width}");
                }
            }
        }
    }

    /// A layout that names no row has no line along one.
    #[test]
    fn cell_lines_refuse_an_undefined_layout() {
        for width in [1, 4] {
            assert_eq!(of(&rdna4(), MatrixLayout::Undefined, width), None);
        }
        assert_eq!(widest(&rdna4(), MatrixLayout::Undefined), 1);
    }

    /// A width that is not a power of two is no line the regroup builds.
    #[test]
    fn a_width_off_a_power_of_two_is_refused() {
        assert_eq!(of(&rdna4(), MatrixLayout::RowMajor, 3), None);
    }
}

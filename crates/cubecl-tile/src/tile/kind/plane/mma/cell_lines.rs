//! How a plane's units write the cells of a sum they hold as whole lines of its destination
//! ([`CellLines`]), regrouping them across units in registers where no unit holds a line.

use cubecl::cmma::{MatrixIdent, MatrixLayout, MmaDefinition};
use cubecl::ir::ElementPositions;
use cubecl::prelude::*;

use super::base::laid_out;
use super::widths::NA;
use crate::*;

/// How the units of a plane write the cells of a sum they hold as lines of `width` cells along a
/// row of their destination: each unit `width` of its cells a line as it holds them, or, where
/// `width` units each hold one cell of every line of a block of them, after a regroup in
/// registers that hands each unit whole lines of that block ([`regroup_cells`]).
///
/// # Guarantees
///
/// - After the regroup, the unit `holders * g + p` holds, at its cells `width * b..width * (b +
///   1)`, the line starting at the position of the `width * b + p`-th cell of unit `holders * g`,
///   its cells in order along the row.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellLines {
    /// Cells a line holds.
    pub width: usize,
    /// The units that hold a line between them as the sum lies: `1` where a unit holds it whole,
    /// `width` where each holds one of its cells.
    pub holders: usize,
}

impl CellLines {
    /// How `lanes` units each holding `elems` cells where `positions` places them write them as
    /// lines `width` cells wide along the rows of a destination that lays the sum out as `layout`,
    /// or `None` where neither a unit nor `width` units together hold a line.
    pub fn of(
        positions: &ElementPositions,
        lanes: usize,
        elems: usize,
        layout: MatrixLayout,
        width: usize,
    ) -> Option<Self> {
        let fits =
            width.is_power_of_two() && elems.is_multiple_of(width) && lanes.is_multiple_of(width);
        let laid = |lane: usize, nth: usize| laid_position(positions.of_nth(lane, nth), layout);
        let starts = |step: usize, count: usize| (0..count).step_by(step);
        let held = fits
            && (0..lanes).all(|lane| {
                starts(width, elems).all(|first| {
                    let (row, col) = laid(lane, first);
                    col.is_multiple_of(width)
                        && (0..width).all(|e| laid(lane, first + e) == (row, col + e))
                })
            });
        let shared = fits
            && starts(width, lanes).all(|lane| {
                starts(width, elems).all(|first| {
                    let col = laid(lane, first).1;
                    col.is_multiple_of(width)
                        && (0..width).all(|e| {
                            let row = laid(lane, first + e).0;
                            (0..width).all(|a| laid(lane + a, first + e) == (row, col + a))
                        })
                })
            });
        match (held, shared) {
            (true, _) => Some(Self { width, holders: 1 }),
            (false, true) => Some(Self {
                width,
                holders: width,
            }),
            (false, false) => None,
        }
    }

    /// The widest line, along the rows of a destination laid out as `layout`, that `lanes` units
    /// each holding `elems` cells where `positions` places them write ([`of`](Self::of)): one
    /// cell where none wider is.
    pub fn widest(
        positions: &ElementPositions,
        lanes: usize,
        elems: usize,
        layout: MatrixLayout,
    ) -> usize {
        (0..=elems.max(1).ilog2())
            .map(|bits| 1usize << bits)
            .filter(|&width| Self::of(positions, lanes, elems, layout, width).is_some())
            .max()
            .unwrap_or(1)
    }
}

/// `(row, col)` of a sum as a destination laid out as `layout` holds it.
fn laid_position((row, col): (usize, usize), layout: MatrixLayout) -> (usize, usize) {
    match layout {
        MatrixLayout::ColMajor => (col, row),
        MatrixLayout::RowMajor | MatrixLayout::Undefined => (row, col),
    }
}

/// Each unit's accumulator cells of `def`, held in `fragment`, written through `mem`'s own write
/// ([`write_cells`]) a line of `mem`'s width at a time, regrouped across units first where no
/// unit holds a line ([`CellLines`]): as a sink that adds whole lines takes them.
///
/// # Panics
///
/// - Where the target states no positions of the sum's cells, or neither a unit nor a line's
///   width of units hold its lines.
#[cube]
pub(crate) fn store_cells<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
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
    let vector_size = def.vector_size(MatrixIdent::Accumulator);
    let positions = def.element_positions(MatrixIdent::Accumulator);
    let lines = comptime!(
        positions
            .and_then(|positions| CellLines::of(&positions, lanes as usize, elems, layout, width))
            .unwrap_or_else(|| panic!(
                "store_cells: a sum drains out of its units' registers where its units, alone or \
                 {width} together, hold lines {width} cells wide along its rows"
            ))
    );
    let mut cells = Array::<T>::new(elems);
    #[unroll]
    for i in 0..comptime!(elems / vector_size) {
        #[unroll]
        for e in 0..vector_size {
            cells[i * vector_size + e] = fragment[i].extract(e);
        }
    }
    regroup_cells::<T>(&mut cells, elems, comptime!(lines.holders));
    let holders = comptime!(lines.holders as u32);
    let first_holder = UNIT_POS_PLANE / holders * holders;
    let place = UNIT_POS_PLANE % holders;
    let size!(W) = width;
    write_cells::<Out, W>(mem, space, |sink| {
        #[unroll]
        for line in 0..comptime!(elems / width) {
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
    });
}

/// Transposes each block of `holders` cells among each `holders` neighbouring units in the plane,
/// in registers: afterwards unit `p` of a group holds, at cell `e` of a block, what unit `e` held
/// at its cell `p`. A butterfly: each step swaps half a block's cells with the unit whose place
/// differs in one bit, through a plane shuffle, with no shared memory and no barrier.
#[cube]
fn regroup_cells<T: Numeric>(
    cells: &mut Array<T>,
    #[comptime] elems: usize,
    #[comptime] holders: usize,
) {
    #[unroll]
    for bit in 0..comptime!(holders.trailing_zeros() as usize) {
        let mask = comptime!(1usize << bit);
        let upper = (UNIT_POS_PLANE / comptime!(mask as u32)) % 2 == 1;
        #[unroll]
        for pair in 0..comptime!(elems / 2) {
            let low = comptime!((pair / mask) * 2 * mask + pair % mask);
            let high = comptime!(low + mask);
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
    use cubecl::ir::{CoordinatePart, ElementIndex};

    /// RDNA3's `f32` accumulator: a lane holds one column, every other row.
    fn rdna3() -> ElementPositions {
        let lane = CoordinatePart::of(ElementIndex::Lane);
        let nth = CoordinatePart::of(ElementIndex::Nth);
        ElementPositions {
            row: vec![nth.spaced_by(2), lane.in_runs_of(16)],
            col: vec![lane.counted_over(16)],
        }
    }

    /// RDNA4's `f32` accumulator: a lane holds eight cells of one row.
    fn rdna4() -> ElementPositions {
        let lane = CoordinatePart::of(ElementIndex::Lane);
        let nth = CoordinatePart::of(ElementIndex::Nth);
        ElementPositions {
            row: vec![lane.counted_over(16)],
            col: vec![nth, lane.in_runs_of(16).spaced_by(8)],
        }
    }

    /// A lane that holds a run along a row writes lines that split it as it holds them.
    #[test]
    fn a_lane_holding_a_run_writes_it_as_it_lies() {
        for width in [1, 2, 4, 8] {
            assert_eq!(
                CellLines::of(&rdna4(), 32, 8, MatrixLayout::RowMajor, width),
                Some(CellLines { width, holders: 1 })
            );
        }
        assert_eq!(
            CellLines::widest(&rdna4(), 32, 8, MatrixLayout::RowMajor),
            8
        );
    }

    /// Lanes that each hold one cell of a row regroup a line among `width` of them.
    #[test]
    fn lanes_holding_a_column_regroup_a_line_among_them() {
        assert_eq!(
            CellLines::of(&rdna3(), 32, 8, MatrixLayout::RowMajor, 4),
            Some(CellLines {
                width: 4,
                holders: 4
            })
        );
        assert_eq!(
            CellLines::widest(&rdna3(), 32, 8, MatrixLayout::RowMajor),
            8
        );
    }

    /// A line wider than a lane's cells, or than the lanes that share its rows, is not written.
    #[test]
    fn a_line_no_lanes_hold_is_refused() {
        assert_eq!(
            CellLines::of(&rdna3(), 32, 8, MatrixLayout::RowMajor, 16),
            None
        );
        assert_eq!(
            CellLines::of(&rdna4(), 32, 8, MatrixLayout::RowMajor, 16),
            None
        );
    }

    /// The line a regrouped unit writes holds, in order, the cells the guarantee names: a mirror
    /// of the butterfly on the host.
    #[test]
    fn a_regrouped_line_holds_its_cells_in_order() {
        let positions = rdna3();
        let lines = CellLines::of(&positions, 32, 8, MatrixLayout::RowMajor, 4).unwrap();
        let mut held: Vec<Vec<(usize, usize)>> = (0..32)
            .map(|lane| (0..8).map(|nth| positions.of_nth(lane, nth)).collect())
            .collect();
        for bit in 0..lines.holders.trailing_zeros() {
            let mask = 1 << bit;
            let before = held.clone();
            for (lane, cells) in held.iter_mut().enumerate() {
                let upper = (lane / mask) % 2 == 1;
                for pair in 0..4 {
                    let low = (pair / mask) * 2 * mask + pair % mask;
                    let high = low + mask;
                    let sent_by_partner = match !upper {
                        true => before[lane ^ mask][low],
                        false => before[lane ^ mask][high],
                    };
                    match upper {
                        true => cells[low] = sent_by_partner,
                        false => cells[high] = sent_by_partner,
                    }
                }
            }
        }
        for (lane, cells) in held.iter().enumerate() {
            for line in 0..2 {
                let first = lane / 4 * 4;
                let (row, col) = positions.of_nth(first, line * 4 + lane % 4);
                let want: Vec<_> = (0..4).map(|e| (row, col + e)).collect();
                assert_eq!(cells[line * 4..line * 4 + 4], want[..], "lane {lane}");
            }
        }
    }
}

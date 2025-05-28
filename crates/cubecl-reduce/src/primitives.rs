use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::BoundChecksInner;
use crate::LineMode;
use crate::ReduceParams;
use crate::instructions::*;
use crate::precision::ReducePrecision;

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReduceRange {
    pub index_start: u32,
    pub index_step: u32,
    pub coordinate_start: u32,
    pub coordinate_end: u32,
    pub coordinate_step: u32,
}

#[cube]
impl ReduceRange {
    pub(crate) fn new<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        match comptime!(params.line_mode) {
            LineMode::Parallel => {
                Self::new_parallel::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
            LineMode::Perpendicular => {
                Self::new_perpendicular::<P, Out>(reduce_index, input, output, axis_reduce, params)
            }
        }
    }

    fn new_parallel<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let shape_axis = input.shape(axis_reduce);

        let mut index_start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            index_start += coordinate * input.stride(axis);
        }
        index_start /= params.line_size_input;

        let coordinate_end = shape_axis;

        let coordinate_step = if params.shared.is_some() {
            CUBE_DIM * params.line_size_input
        } else if params.use_planes {
            CUBE_DIM_X * params.line_size_input
        } else {
            params.line_size_input.runtime()
        };

        ReduceRange {
            index_start,
            index_step: 1,
            coordinate_start: 0,
            coordinate_end,
            coordinate_step,
        }
    }

    fn new_perpendicular<P: ReducePrecision, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] params: ReduceParams,
    ) -> ReduceRange {
        let shape_axis = input.shape(axis_reduce);

        let mut index_start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index * params.line_size_input, axis);
            index_start += coordinate * input.stride(axis);
        }
        index_start /= params.line_size_input;

        let index_step = input.stride(axis_reduce) / params.line_size_input;

        let coordinate_end = shape_axis;

        let coordinate_step = if params.shared.is_some() {
            CUBE_DIM
        } else if params.use_planes {
            CUBE_DIM_X
        } else {
            1_u32.runtime()
        };

        ReduceRange {
            index_start,
            index_step,
            coordinate_start: 0,
            coordinate_step,
            coordinate_end,
        }
    }
}

/// Use an individual unit to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive).
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// Since each individual unit performs a reduction, this function is meant to be called
/// with either a different `items` for each unit, a different `range` or both based on ABSOLUTE_UNIT_POS.
#[cube]
pub fn reduce_slice<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    range: ReduceRange,
    inst: &R,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(inst, line_size);

    let mut index = range.index_start;
    for coordinate in range_stepped(
        range.coordinate_start,
        range.coordinate_end,
        range.coordinate_step,
    ) {
        let requirements = R::requirements(inst);
        let coordinates = if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(coordinate, line_size, line_mode))
        } else {
            ReduceCoordinate::new_NotRequired()
        };
        reduce_inplace::<P, R>(
            inst,
            &mut accumulator,
            items.read(index),
            coordinates,
            false,
        );
        index += range.index_step;
    }

    accumulator
}

/// Use an individual plane  to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive).
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// This assumes that `UNIT_POS_X` provides the index of unit with a plane and that `CUBE_DIM_X` is the plane dimension.
/// That is, the cube_dim is `CubeDim::new_2d(plane_dim, plane_count)`.
///
/// Since each individual plane performs a reduction, this function is meant to be called
/// with either a different `items` for each plane, a different `range` or both based on
/// the absolute plane position (`CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y`).
#[cube]
pub fn reduce_slice_plane<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] bound_checks: BoundChecksInner,
) -> R::AccumulatorItem {
    let plane_dim = CUBE_DIM_X;

    let mut accumulator = R::null_accumulator(inst, line_size);

    let mut first_index = range.index_start;
    for first_coordinate in range_stepped(
        range.coordinate_start,
        range.coordinate_end,
        range.coordinate_step,
    ) {
        let unit_coordinate_offset = match line_mode {
            LineMode::Parallel => UNIT_POS_X * line_size,
            LineMode::Perpendicular => UNIT_POS_X,
        };
        let unit_coordinate = first_coordinate + unit_coordinate_offset;

        let requirements = R::requirements(inst);
        let coordinates = if comptime![requirements.coordinates] {
            ReduceCoordinate::new_Required(fill_coordinate_line(
                unit_coordinate,
                line_size,
                line_mode,
            ))
        } else {
            ReduceCoordinate::new_NotRequired()
        };

        let index = first_index + UNIT_POS_X * range.index_step;
        let item = match bound_checks {
            BoundChecksInner::None => items.read(index),
            BoundChecksInner::Mask => {
                let mask = unit_coordinate < range.coordinate_end;
                let index = index * u32::cast_from(mask);
                select(mask, items.read(index), R::null_input(inst, line_size))
            }
            BoundChecksInner::Branch => {
                if unit_coordinate < range.coordinate_end {
                    items.read(index)
                } else {
                    R::null_input(inst, line_size)
                }
            }
        };

        reduce_inplace::<P, R>(inst, &mut accumulator, item, coordinates, true);

        first_index += plane_dim * range.index_step;
    }
    accumulator
}

/// Use an individual cube to reduce the `items` with the specified range.
/// That is, this will reduces `items[range.start]`, `items[range.start + range.step]`
/// until `items[range.end]` (exclusive). Inside a cube, the reduction will use plane operations
/// if `use_planes` is set to `true`.
///
/// This reduces using the given `line_mode` but doesn't reduce the accumulator itself.
///
/// When `use_planes` is `true`, this assumes that `UNIT_POS_Y` provides the relative position
/// of a plane within its cube.
///
/// Since each individual cube performs a reduction, this function is meant to be called
/// with either a different `items` for each cube, a different `range` or both based on `CUBE_POS`.
#[cube]
pub fn reduce_slice_shared<P: ReducePrecision, I: List<Line<P::EI>>, R: ReduceInstruction<P>>(
    items: &I,
    inst: &R,
    range: ReduceRange,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] use_planes: bool,
    #[comptime] bound_checks: BoundChecksInner,
) -> R::SharedAccumulator {
    // The index used to read and write into the accumulator.
    let accumulator_index = if use_planes { UNIT_POS_Y } else { UNIT_POS };

    let requirements = R::requirements(inst);
    let mut accumulator =
        R::SharedAccumulator::allocate(accumulator_size, line_size, requirements.coordinates);

    R::SharedAccumulator::write(
        &mut accumulator,
        accumulator_index,
        R::null_accumulator(inst, line_size),
    );

    let mut first_index = range.index_start;
    for first_coordinate in range_stepped(
        range.coordinate_start,
        range.coordinate_end,
        range.coordinate_step,
    ) {
        let unit_coordinate_offset = match line_mode {
            LineMode::Parallel => UNIT_POS * line_size,
            LineMode::Perpendicular => UNIT_POS,
        };
        let unit_coordinate = first_coordinate + unit_coordinate_offset;

        let index = first_index + UNIT_POS * range.index_step;

        let item = match bound_checks {
            BoundChecksInner::None => items.read(index),
            BoundChecksInner::Mask => {
                let mask = unit_coordinate < range.coordinate_end;
                let index = index * u32::cast_from(mask);
                select(mask, items.read(index), R::null_input(inst, line_size))
            }
            BoundChecksInner::Branch => {
                if unit_coordinate < range.coordinate_end {
                    items.read(index)
                } else {
                    R::null_input(inst, line_size)
                }
            }
        };

        let coordinates = if comptime! {requirements.coordinates} {
            let coordinate = fill_coordinate_line(unit_coordinate, line_size, line_mode);
            let coordinate = select(
                unit_coordinate < range.coordinate_end,
                coordinate,
                Line::empty(line_size).fill(u32::MAX),
            );

            ReduceCoordinate::new_Required(coordinate)
        } else {
            ReduceCoordinate::new_NotRequired()
        };

        reduce_shared_inplace::<P, R>(
            inst,
            &mut accumulator,
            accumulator_index,
            item,
            coordinates,
            use_planes,
        );
        first_index += range.index_step * CUBE_DIM;
    }
    accumulator
}

// If line mode is parallel, fill a line with `x, x+1, ... x+ line_size - 1` where `x = first`.
// If line mode is perpendicular, fill a line with `x, x, ... x` where `x = first`.
#[cube]
fn fill_coordinate_line(
    first: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Line<u32> {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let mut coordinates = Line::empty(line_size);
            #[unroll]
            for j in 0..line_size {
                coordinates[j] = first + j;
            }
            coordinates
        }
        LineMode::Perpendicular => Line::empty(line_size).fill(first),
    }
}

/// Use all units within a cube to fuse the first `size` elements of `accumulator` inplace like this with some padding if `size` is not a power of 2.
///
///
/// ```ignored
///
///     0   1   2   3   4   5   6   7
///     |   |   |   |   |   |   |   |
///     +---+   +---+   +---+   +---+
///     |       |       |       |
///     +-------+       +-------+
///     |               |
///     +---------------+
///     |
///     *
///
/// ```
///
/// The outcome is stored in the first element of the accumulator and also returned by this function for convenience.
///
/// Since each individual cube performs a reduction, this function is meant to be called
/// with a different `accumulator` for each cube based on `CUBE_POS`.
///
/// There is no out-of-bound check, so it is the responsibility of the caller to ensure that `size` is at most the length
/// of the shared memory and that there are at least `size` units within each cube.
#[cube]
pub fn reduce_tree<P: ReducePrecision, Inst: ReduceInstruction<P>>(
    inst: &Inst,
    accumulator: &mut Inst::SharedAccumulator,
    #[comptime] size: u32,
) -> Inst::AccumulatorItem {
    if comptime!(size.is_power_of_two()) {
        let mut num_active_units = size.runtime();
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_active_units {
                fuse_accumulator_inplace::<P, Inst>(inst, accumulator, destination, origin);
            }
            jump *= 2;
            sync_cube();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_remaining_items / 2 {
                fuse_accumulator_inplace::<P, Inst>(inst, accumulator, destination, origin);
            }
            num_remaining_items = div_ceil(num_remaining_items, 2);
            jump *= 2;
            sync_cube();
        }
    }
    sync_cube();
    Inst::SharedAccumulator::read(accumulator, 0)
}

#[cube]
#[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

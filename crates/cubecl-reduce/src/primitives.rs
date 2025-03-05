use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::instructions::*;
use crate::LineMode;

/// A simple range to specify how to iterate a slice when performing a reduction.
#[derive(CubeType)]
pub struct ReduceRange {
    pub start: u32,
    pub end: u32,
    pub step: u32,
}

#[cube]
impl ReduceRange {
    pub(crate) fn new<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<In>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] line_size: u32,
        #[comptime] line_mode: LineMode,
    ) -> ReduceRange {
        match comptime!(line_mode) {
            LineMode::Parallel => {
                Self::new_parallel::<In, Out>(reduce_index, input, output, axis_reduce, line_size)
            }
            LineMode::Perpendicular => Self::new_perpendicular::<In, Out>(
                reduce_index,
                input,
                output,
                axis_reduce,
                line_size,
            ),
        }
    }

    fn new_parallel<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<In>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] line_size: u32,
    ) -> ReduceRange {
        let mut start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            start += coordinate * input.stride(axis);
        }
        start /= line_size;

        let end = start + input.shape(axis_reduce) / line_size;
        let end = select(end < input.buffer_len(), end, input.buffer_len());

        ReduceRange {
            start,
            end,
            step: 1,
        }
    }

    fn new_perpendicular<In: Numeric, Out: Numeric>(
        reduce_index: u32,
        input: &VirtualTensor<In>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        axis_reduce: u32,
        #[comptime] line_size: u32,
    ) -> ReduceRange {
        let mut start = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index * line_size, axis);
            start += coordinate * input.stride(axis);
        }
        start /= line_size;

        let step = input.stride(axis_reduce) / line_size;

        let end = start + input.shape(axis_reduce) * step;
        let end = select(end < input.buffer_len(), end, input.buffer_len());

        ReduceRange { start, end, step }
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
pub fn reduce_slice<N: Numeric, R: ReduceInstruction<N>>(
    items: &VirtualTensor<N>,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut index = range.start;
    let mut coordinate = 0;
    while index < range.end {
        let coordinates = fill_coordinate_line(coordinate, line_size, line_mode);
        reduce_inplace::<N, R>(&mut accumulator, items.read(index), coordinates, false);
        index += range.step;
        coordinate += 1;
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
pub fn reduce_slice_plane<N: Numeric, R: ReduceInstruction<N>>(
    items: &VirtualTensor<N>,
    range: ReduceRange,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut first_index = range.start;
    let mut first_coordinate = 0;
    while first_index < range.end {
        let coordinates = fill_coordinate_line(first_coordinate + UNIT_POS_X, line_size, line_mode);

        let index = first_index + UNIT_POS_X * range.step;
        let item = select(
            index < range.end,
            items.read(index),
            R::null_input(line_size),
        );

        reduce_inplace::<N, R>(&mut accumulator, item, coordinates, true);

        let plane_dim = CUBE_DIM_X;
        first_index += plane_dim * range.step;
        first_coordinate += plane_dim;
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
pub fn reduce_slice_shared<N: Numeric, R: ReduceInstruction<N>>(
    items: &VirtualTensor<N>,
    range: ReduceRange,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] use_planes: bool,
) -> R::SharedAccumulator {
    // The index used to read and write into the accumulator.
    let accumulator_index = if use_planes { UNIT_POS_Y } else { UNIT_POS };

    let mut accumulator = R::SharedAccumulator::allocate(accumulator_size, line_size);

    R::SharedAccumulator::write(
        &mut accumulator,
        accumulator_index,
        R::null_accumulator(line_size),
    );

    let mut first_index = range.start;
    let mut first_coordinate = 0;
    while first_index < range.end {
        let index = first_index + UNIT_POS * range.step;
        let item = select(
            index < range.end,
            items.read(index),
            R::null_input(line_size),
        );
        let coordinate = fill_coordinate_line(first_coordinate + UNIT_POS, line_size, line_mode);
        let coordinate = select(
            index < range.end,
            coordinate,
            Line::empty(line_size).fill(u32::MAX),
        );
        reduce_shared_inplace::<N, R>(
            &mut accumulator,
            accumulator_index,
            item,
            coordinate,
            use_planes,
        );
        first_index += range.step * CUBE_DIM;
        first_coordinate += CUBE_DIM;
    }
    accumulator
}

// If line mode is parallel, fill a line with `x, x+1, ... x+ line_size - 1` where `x = first * line_size`.
// If line mode is perpendicular, fill a line with `x, x, ... x` where `x = first`.
#[cube]
fn fill_coordinate_line(
    first: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Line<u32> {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let mut coordinates = Line::empty(line_size).fill(first * line_size);
            if line_size > 1 {
                #[unroll]
                for j in 0..line_size {
                    coordinates[j] += j;
                }
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
/// There is no out-of-bound check, so it is the responsability of the caller to ensure that `size` is at most the length
/// of the shared memory and that there are at least `size` units within each cube.
#[cube]
pub fn reduce_tree<In: Numeric, Inst: ReduceInstruction<In>>(
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
                fuse_accumulator_inplace::<In, Inst>(accumulator, destination, origin);
            }
            jump *= 2;
            sync_units();
        }
    } else {
        let mut num_remaining_items = size.runtime();
        let mut jump = 1;
        while num_remaining_items > 1 {
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_remaining_items / 2 {
                fuse_accumulator_inplace::<In, Inst>(accumulator, destination, origin);
            }
            num_remaining_items = div_ceil(num_remaining_items, 2);
            jump *= 2;
            sync_units();
        }
    }
    sync_units();
    Inst::SharedAccumulator::read(accumulator, 0)
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

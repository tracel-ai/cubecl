use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{ReduceArgMax, ReduceArgMin, ReduceMean, ReduceProd, ReduceSum};

/// An instruction for the [reduce_shared](reduce_shared) algorithm.
#[cube]
pub trait ReduceSharedInstruction<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator.
    type Accumulator: CubeType;

    fn create_accumulator(#[comptime] length: u32, #[comptime] line_size: u32)
        -> Self::Accumulator;

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    );

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        coordinate: u32,
    );

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32);

    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    );
}

#[cube]
pub fn reduce_shared<RD: ReduceSharedInstruction<EI>, EI: Numeric, EO: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<EO>>,
    reduce_dim: u32,
    #[comptime] cube_dim: u32,
    #[comptime] exact_shape: bool,
) {
    let line_size = input.line_size();
    let mut accumulator = RD::create_accumulator(cube_dim, line_size);
    RD::init_accumulator(&mut accumulator, UNIT_POS, line_size);

    // Compute the first index where to start the reduction for the current cube.
    // First, compute the coordinate corresponding to the CUBE_POS element of the output tensor
    // Then, use the strides of the input tensor to find the index of the same coordinate
    // in the input tensor.
    let mut offset = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(CUBE_POS, axis);
        offset += coordinate * input.stride(axis);
    }

    let num_items_per_unit = div_ceil(input.shape(reduce_dim), cube_dim, exact_shape);
    // TODO: Change ordering of reduce within cube.
    offset += UNIT_POS * num_items_per_unit * input.stride(reduce_dim);

    for i in 0..num_items_per_unit {
        let index = offset + i * input.stride(reduce_dim);
        let coordinate = i + num_items_per_unit * UNIT_POS;
        #[allow(clippy::collapsible_else_if)]
        if exact_shape {
            RD::accumulate(&mut accumulator, UNIT_POS, input[index], coordinate);
        } else {
            if coordinate < input.shape(reduce_dim) {
                RD::accumulate(&mut accumulator, UNIT_POS, input[index], coordinate);
            }
        }
    }

    sync_units();

    // Merge the accumulator like this with some padding if CUBE_DIM is not a power of 2.
    //
    // 0   1   2   3   4   5   6   7
    // |   |   |   |   |   |   |   |
    // +-+-+   +-+-+   +-+-+   +-+-+
    //   |       |       |       |
    //   +---+---+       +---+---+
    //       |               |
    //       +-------+-------+
    //               |
    //               *
    //
    // Be careful if you want to change the ordering as some reduction algorithms
    // like ArgMax relies on that to be associative without being commutative.
    // In that case, it allows ArgMax to always return the smallest coordinate when there are
    // multiple items with the maximum value without having to explicitely check the coordinates
    // within each call to RD::merge.
    if comptime!(cube_dim.is_power_of_two()) {
        let mut num_active_units = cube_dim;
        let mut jump = 1;
        while num_active_units > 1 {
            num_active_units /= 2;
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_active_units {
                RD::merge(&mut accumulator, destination, origin);
            }
            jump *= 2;
            sync_units();
        }
    } else {
        let mut num_remaining_items = cube_dim;
        let mut jump = 1;
        while num_remaining_items > 1 {
            let num_active_units = div_ceil(num_remaining_items, 2, false);
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if origin < num_remaining_items {
                RD::merge(&mut accumulator, destination, origin);
            }
            num_remaining_items = num_active_units;
            jump *= 2;
            sync_units();
        }
    }

    if UNIT_POS == 0 {
        RD::write_first(output, &accumulator, CUBE_POS, input.shape(reduce_dim));
    }
}

#[cube]
fn div_ceil(a: u32, b: u32, #[comptime] exact: bool) -> u32 {
    if exact {
        a / b
    } else {
        (a + b - 1) / b
    }
}

// Implementations for common instructions.

#[cube]
impl<EI: Numeric> ReduceSharedInstruction<EI> for ReduceSum {
    type Accumulator = SharedMemory<Line<EI>>;

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        SharedMemory::new_lined(length, line_size)
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        accumulator[index] = Line::empty(line_size).fill(EI::from_int(0));
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        _coordinate: u32,
    ) {
        accumulator[destination] += item;
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let item = accumulator[origin];
        accumulator[destination] += item;
    }

    /// Write the result of the reduction stored in `accumulator` into `output[index]`.
    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator[0]);
    }
}

#[cube]
impl<EI: Numeric> ReduceSharedInstruction<EI> for ReduceProd {
    type Accumulator = SharedMemory<Line<EI>>;

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        SharedMemory::new_lined(length, line_size)
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        accumulator[index] = Line::empty(line_size).fill(EI::from_int(1));
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        _coordinate: u32,
    ) {
        accumulator[destination] *= item;
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let item = accumulator[origin];
        accumulator[destination] *= item;
    }

    /// Write the result of the reduction stored in `accumulator` into `output[index]`.
    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator[0]);
    }
}

#[cube]
impl<EI: Numeric> ReduceSharedInstruction<EI> for ReduceMean {
    type Accumulator = SharedMemory<Line<EI>>;

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        SharedMemory::new_lined(length, line_size)
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        accumulator[index] = Line::empty(line_size).fill(EI::from_int(0));
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        _coordinate: u32,
    ) {
        accumulator[destination] += item;
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let item = accumulator[origin];
        accumulator[destination] += item;
    }

    /// Write the result of the reduction stored in `accumulator` into `output[index]`.
    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator[0])
            / Line::empty(output.line_size()).fill(EO::cast_from(shape_reduce_dim));
    }
}

#[cube]
impl<EI: Numeric> ReduceSharedInstruction<EI> for ReduceArgMax {
    type Accumulator = (SharedMemory<Line<EI>>, SharedMemory<Line<u32>>);

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        (
            SharedMemory::new_lined(length, line_size),
            SharedMemory::new_lined(length, line_size),
        )
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        let (items, indices) = accumulator;
        items[index] = Line::empty(line_size).fill(EI::MIN);
        indices[index] = Line::empty(line_size).fill(0);
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        coordinate: u32,
    ) {
        let (currents, indices) = accumulator;
        let to_replace = item.greater_than(currents[destination]);
        currents[destination] = select_many(to_replace, item, currents[destination]);
        indices[destination] = select_many(
            to_replace,
            Line::empty(indices[destination].size()).fill(coordinate),
            indices[destination],
        );
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let (items, indices) = accumulator;
        let to_replace = items[origin].greater_than(items[destination]);
        items[destination] = select_many(to_replace, items[origin], items[destination]);
        indices[destination] = select_many(to_replace, indices[origin], indices[destination]);
    }

    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator.1[0]);
    }
}

#[cube]
impl<EI: Numeric> ReduceSharedInstruction<EI> for ReduceArgMin {
    type Accumulator = (SharedMemory<Line<EI>>, SharedMemory<Line<u32>>);

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        (
            SharedMemory::new_lined(length, line_size),
            SharedMemory::new_lined(length, line_size),
        )
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        let (items, indices) = accumulator;
        items[index] = Line::empty(line_size).fill(EI::MAX);
        indices[index] = Line::empty(line_size).fill(0);
    }

    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        coordinate: u32,
    ) {
        let (currents, indices) = accumulator;
        let to_replace = item.less_than(currents[destination]);
        currents[destination] = select_many(to_replace, item, currents[destination]);
        indices[destination] = select_many(
            to_replace,
            Line::empty(indices[destination].size()).fill(coordinate),
            indices[destination],
        );
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let (items, indices) = accumulator;
        let to_replace = items[origin].less_than(items[destination]);
        items[destination] = select_many(to_replace, items[origin], items[destination]);
        indices[destination] = select_many(to_replace, indices[origin], indices[destination]);
    }

    fn write_first<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: &Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator.1[0]);
    }
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{ReduceArgMax, ReduceArgMin, ReduceMean, ReduceProd, ReduceSum};

/// An instruction for the [reduce_shared](reduce_shared) algorithm.
#[cube]
pub trait ReduceSharedInstruction<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator.
    /// The implement works on lines. Most likely, the accumulator is Line<T>
    /// for some CubePrimitive type T instead of simply T.
    type Accumulator: CubeType;

    /// Create an unitialized accumulator containing length item of the given line_size.
    fn create_accumulator(#[comptime] length: u32, #[comptime] line_size: u32)
        -> Self::Accumulator;

    /// Set the null value of the reduction into the accumulator at the given index..
    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    );

    // Reduce item and coordinate into the accumulator at the index destination.
    fn accumulate(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<EI>,
        coordinate: u32,
    );

    // Reduce the items at destination and origin within the accumulator and store the result at the destination.
    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32);

    // Write the first item of accumulator into the ouput at the given index.
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

    let num_items_per_unit = if exact_shape {
        input.shape(reduce_dim) / cube_dim
    } else {
        div_ceil(input.shape(reduce_dim), cube_dim)
    };

    // The unit at UNIT_POS reduces the items with coordinate (UNIT_POS, UNIT_POS + cube_dim, UNIT_POS + 2 * cube_dim, ...).
    // The result in stored into the accumulator at index UNIT_POS.
    for i in 0..num_items_per_unit {
        let coordinate = i * cube_dim + UNIT_POS;
        let index = offset + coordinate * input.stride(reduce_dim);
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
            let destination = jump * 2 * UNIT_POS;
            let origin = jump * (2 * UNIT_POS + 1);
            if UNIT_POS < num_remaining_items / 2 {
                RD::merge(&mut accumulator, destination, origin);
            }
            num_remaining_items = div_ceil(num_remaining_items, 2);
            jump *= 2;
            sync_units();
        }
    }

    if UNIT_POS == 0 {
        RD::write_first(output, &accumulator, CUBE_POS, input.shape(reduce_dim));
    }
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
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
        let (acc_items, acc_coordinates) = accumulator;
        let coordinates = Line::empty(acc_items[destination].size()).fill(coordinate);

        let (new_items, new_coordinates) = Self::choose_argmax(
            acc_items[destination],
            acc_coordinates[destination],
            item,
            coordinates,
        );

        acc_items[destination] = new_items;
        acc_coordinates[destination] = new_coordinates;
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let (items, coordinates) = accumulator;

        let (new_items, new_coordinates) = Self::choose_argmax(
            items[destination],
            coordinates[destination],
            items[origin],
            coordinates[origin],
        );

        items[destination] = new_items;
        coordinates[destination] = new_coordinates;
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
        let (acc_items, acc_coordinates) = accumulator;
        let coordinates = Line::empty(acc_items[destination].size()).fill(coordinate);

        let (new_items, new_coordinates) = Self::choose_argmin(
            acc_items[destination],
            acc_coordinates[destination],
            item,
            coordinates,
        );

        acc_items[destination] = new_items;
        acc_coordinates[destination] = new_coordinates;
    }

    fn merge(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        let (items, coordinates) = accumulator;

        let (new_items, new_coordinates) = Self::choose_argmin(
            items[destination],
            coordinates[destination],
            items[origin],
            coordinates[origin],
        );

        items[destination] = new_items;
        coordinates[destination] = new_coordinates;
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

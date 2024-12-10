use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{
    fuse_accumulator_inplace, reduce_inplace, reduce_shared_inplace, LineMode, Reduce,
    ReduceConfig, ReduceStrategy, SharedAccumulator,
};

/// Entry point for reduce.
#[allow(clippy::too_many_arguments)]
pub fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: Reduce<In>>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    match (strategy.use_planes, strategy.shared, config.line_mode) {
        (false, false, LineMode::Parallel) => unsafe {
            kernel_reduce_unit_parallel::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.line_size,
                config.bound_checks,
            )
        },
        (false, false, LineMode::Perpendicular) => unsafe {
            kernel_reduce_unit_perpendicular::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.line_size,
                config.bound_checks,
            )
        },
        (true, false, LineMode::Parallel) => unsafe {
            kernel_reduce_plane_parallel::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.line_size,
                config.bound_checks,
            )
        },
        (true, false, LineMode::Perpendicular) => unsafe {
            kernel_reduce_plane_perpendicular::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.line_size,
                config.bound_checks,
            )
        },
        (false, true, LineMode::Parallel) => unsafe {
            kernel_reduce_shared_parallel::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.cube_dim.num_elems(),
                config.line_size,
                config.bound_checks,
            )
        },
        (false, true, LineMode::Perpendicular) => unsafe {
            kernel_reduce_shared_perpendicular::launch_unchecked::<In, Out, Rd, Run>(
                client,
                config.cube_count,
                config.cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.cube_dim.num_elems(),
                config.line_size,
                config.bound_checks,
            )
        },
        _ => unimplemented!(),
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_unit_parallel<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    if bound_checks && ABSOLUTE_POS >= output.len() {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(ABSOLUTE_POS, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size);
    let shape = div_ceil(input.shape(axis_reduce), line_size);

    let out = reduce_slice::<In, R>(
        input.to_slice(),
        offset,
        offset + shape * stride,
        stride,
        line_size,
        LineMode::Parallel,
    );
    output[ABSOLUTE_POS] = R::merge_line::<Out>(out, input.shape(axis_reduce))
}

#[cube(launch_unchecked)]
fn kernel_reduce_unit_perpendicular<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    let num_active_units = output.len() / line_size;

    if bound_checks && ABSOLUTE_POS >= num_active_units {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(ABSOLUTE_POS * line_size, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size);
    let shape = input.shape(axis_reduce);

    let out = reduce_slice::<In, R>(
        input.to_slice(),
        offset,
        offset + shape * stride,
        stride,
        line_size,
        LineMode::Perpendicular,
    );

    let out = R::to_output_perpendicular(out, input.shape(axis_reduce));

    #[unroll]
    for k in 0..line_size {
        output[line_size * ABSOLUTE_POS + k] = out[k];
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_plane_parallel<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    let plane_count_per_cube = CUBE_DIM_Y;
    let plane_pos = plane_count_per_cube * CUBE_POS + UNIT_POS_Y;

    if bound_checks && plane_pos >= output.len() {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(plane_pos, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size); // 1
    let shape = div_ceil(input.shape(axis_reduce), line_size);

    let end = offset + shape * stride;
    let out = reduce_slice_plane::<In, R>(
        input.to_slice(),
        offset,
        select(end < input.len(), end, input.len()),
        stride,
        line_size,
        LineMode::Parallel,
    );
    output[plane_pos] = R::merge_line::<Out>(out, input.shape(axis_reduce))
}

#[cube(launch_unchecked)]
fn kernel_reduce_plane_perpendicular<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    let plane_count_per_cube = CUBE_DIM_Y;
    let plane_pos = plane_count_per_cube * CUBE_POS + UNIT_POS_Y;

    if bound_checks && plane_pos >= output.len() {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(plane_pos * line_size, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size);
    let shape = input.shape(axis_reduce);

    let end = offset + shape * stride;
    let out = reduce_slice_plane::<In, R>(
        input.to_slice(),
        offset,
        select(end < input.len(), end, input.len()),
        stride,
        line_size,
        LineMode::Perpendicular,
    );

    let out = R::to_output_perpendicular(out, input.shape(axis_reduce));

    #[unroll]
    for k in 0..line_size {
        output[line_size * plane_pos + k] = out[k];
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_shared_parallel<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    if bound_checks && CUBE_POS >= output.len() {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(CUBE_POS, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size);
    let shape = div_ceil(input.shape(axis_reduce), line_size);

    let end = offset + shape * stride;
    let mut accumulator = reduce_slice_shared::<In, R>(
        input.to_slice(),
        offset,
        select(end < input.len(), end, input.len()),
        stride,
        accumulator_size,
        line_size,
        LineMode::Parallel,
    );

    sync_units();

    let out = reduce_tree::<In, R>(&mut accumulator);

    if UNIT_POS == 0 {
        output[CUBE_POS] = R::merge_line::<Out>(out, input.shape(axis_reduce))
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_shared_perpendicular<In: Numeric, Out: Numeric, R: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    if bound_checks && CUBE_POS >= output.len() / line_size {
        return;
    }

    // Compute the first index where to start the reduction.
    let offset = compute_reduce_offset(CUBE_POS * line_size, input, output, line_size);
    let stride = div_ceil(input.stride(axis_reduce), line_size);
    let shape = input.shape(axis_reduce);

    let end = offset + shape * stride;
    let end = select(end < input.len(), end, input.len());

    let mut accumulator = reduce_slice_shared::<In, R>(
        input.to_slice(),
        offset,
        end,
        stride,
        accumulator_size,
        line_size,
        LineMode::Perpendicular,
    );

    sync_units();

    let out = reduce_tree::<In, R>(&mut accumulator);
    if UNIT_POS == 0 {
        let out = R::to_output_perpendicular(out, input.shape(axis_reduce));

        #[unroll]
        for k in 0..line_size {
            output[line_size * CUBE_POS + k] = out[k];
        }
    }
}

#[cube]
pub fn compute_reduce_offset<In: CubeType, Out: CubeType>(
    index: u32,
    input: &Tensor<In>,
    output: &Tensor<Out>,
    #[comptime] line_size: u32,
) -> u32 {
    let mut offset = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(index, axis);
        offset += coordinate * input.stride(axis);
    }
    offset / line_size
}

#[cube]
pub fn reduce_slice<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut index = start;
    let mut coordinate = 0;
    while index < end {
        let coordinates = match comptime!(line_mode) {
            LineMode::Parallel => {
                let mut coordinates = Line::empty(line_size).fill(coordinate * line_size);
                if line_size > 1 {
                    #[unroll]
                    for j in 0..line_size {
                        coordinates[j] += j;
                    }
                }
                coordinates
            }
            LineMode::Perpendicular => Line::empty(line_size).fill(coordinate),
        };
        reduce_inplace::<N, R>(&mut accumulator, items[index], coordinates, false);
        index += stride;
        coordinate += 1;
    }
    accumulator
}

#[cube]
pub fn reduce_slice_plane<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::AccumulatorItem {
    let mut accumulator = R::null_accumulator(line_size);

    let mut first_index = start;
    let mut first_coordinate = 0;
    while first_index < end {
        let coordinates = match comptime!(line_mode) {
            LineMode::Parallel => {
                let mut coordinates =
                    Line::empty(line_size).fill((first_coordinate + UNIT_POS_X) * line_size);
                if line_size > 1 {
                    #[unroll]
                    for j in 0..line_size {
                        coordinates[j] += j;
                    }
                }
                coordinates
            }
            LineMode::Perpendicular => {
                Line::empty(line_size).fill(first_coordinate + UNIT_POS_X)
            }
        };

        let index = first_index + UNIT_POS_X * stride;
        let item = select(index < end, items[index], R::null_input(line_size));

        reduce_inplace::<N, R>(&mut accumulator, item, coordinates, true);

        let plane_dim = CUBE_DIM_X;
        first_index += plane_dim * stride;
        first_coordinate += plane_dim;
    }
    accumulator
}

#[cube]
pub fn reduce_slice_shared<N: Numeric, R: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] accumulator_size: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> R::SharedAccumulator {
    let mut accumulator = R::SharedAccumulator::allocate(accumulator_size, line_size);
    R::SharedAccumulator::write(&mut accumulator, UNIT_POS, R::null_accumulator(line_size));

    let mut first_index = start;
    let mut first_coordinate = 0;
    while first_index < end {
        let coordinate = first_coordinate + UNIT_POS;
        let line_coordinate = match comptime!(line_mode) {
            LineMode::Parallel => {
                let mut coordinates = Line::empty(line_size).fill(coordinate * line_size);
                if line_size > 1 {
                    #[unroll]
                    for j in 0..line_size {
                        coordinates[j] += j;
                    }
                }
                coordinates
            }
            LineMode::Perpendicular => Line::empty(line_size).fill(coordinate),
        };
        let index = first_index + UNIT_POS * stride;
        let item = select(index < end, items[index], R::null_input(line_size));
        reduce_shared_inplace::<N, R>(&mut accumulator, UNIT_POS, item, line_coordinate, false);
        first_index += stride * CUBE_DIM;
        first_coordinate += CUBE_DIM;
    }
    accumulator
}

/// Use all units within a cube to fuse an accumulator inplace like this with some padding if CUBE_DIM is not a power of 2.
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
/// This assumes that the size of accumulator is exactly CUBE_DIM and leads to undefined behavior if not.
#[cube]
pub fn reduce_tree<In: Numeric, Inst: Reduce<In>>(
    accumulator: &mut Inst::SharedAccumulator,
) -> Inst::AccumulatorItem {
    if comptime!(CUBE_DIM.is_power_of_two()) {
        let mut num_active_units = CUBE_DIM;
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
        let mut num_remaining_items = CUBE_DIM;
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

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{LineMode, Reduce, ReduceConfig, ReduceInstruction, ReduceStrategy};

/// Entry point for reduce.
#[allow(clippy::too_many_arguments)]
pub fn launch_reduce<R: Runtime, In: Numeric, Out: Numeric, Inst: ReduceInstruction<In>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    match (strategy.use_planes, strategy.shared, config.line_mode) {
        (false, false, LineMode::Parallel) => unsafe {
            kernel_reduce_unit_parallel::launch_unchecked::<In, Out, Inst, R>(
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
            kernel_reduce_unit_perpendicular::launch_unchecked::<In, Out, Inst, R>(
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
        (true, false, _) => unsafe {
            kernel_reduce_plane::launch_unchecked::<In, Out, Inst, R>(
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
        _ => unimplemented!(),
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_unit_parallel<In: Numeric, Out: Numeric, Inst: Reduce<In>>(
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

    let out = reduce_slice::<In, Inst>(
        input.to_slice(),
        offset,
        offset + shape * stride,
        stride,
        line_size,
        LineMode::Parallel,
    );
    output[ABSOLUTE_POS] = Inst::merge_line::<Out>(out, input.shape(axis_reduce))
}

#[cube(launch_unchecked)]
fn kernel_reduce_unit_perpendicular<In: Numeric, Out: Numeric, Inst: Reduce<In>>(
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

    let out = reduce_slice::<In, Inst>(
        input.to_slice(),
        offset,
        offset + shape * stride,
        stride,
        line_size,
        LineMode::Perpendicular,
    );

    let out = Inst::to_output_perpendicular(out, input.shape(axis_reduce));

    #[unroll]
    for k in 0..line_size {
        output[line_size * ABSOLUTE_POS + k] = out[k];
    }
}

#[cube(launch_unchecked)]
fn kernel_reduce_plane<In: Numeric, Out: Numeric, Inst: Reduce<In>>(
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
    let out = reduce_slice_plane::<In, Inst>(
        input.to_slice(),
        offset,
        select(end < input.len(), end, input.len()),
        stride,
        line_size,
    );
    output[plane_pos] = Inst::merge_line::<Out>(out, input.shape(axis_reduce))
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
pub fn reduce_slice<N: Numeric, Instr: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
) -> Instr::Accumulator {
    let mut accumulator = Instr::init_accumulator(line_size);

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
        Instr::reduce(&mut accumulator, items[index], coordinates, false);
        index += stride;
        coordinate += 1;
    }
    accumulator
}

#[cube]
pub fn reduce_slice_plane<N: Numeric, Instr: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] line_size: u32,
) -> Instr::Accumulator {
    let mut accumulator = Instr::init_accumulator(line_size);

    let mut first_index = start;
    let mut first_coordinate = 0;
    while first_index < end {
        let mut coordinates =
            Line::empty(line_size).fill((first_coordinate + UNIT_POS_X) * line_size);
        if line_size > 1 {
            #[unroll]
            for j in 0..line_size {
                coordinates[j] += j;
            }
        }

        let index = first_index + UNIT_POS_X * stride;
        let item = select(
            index < end,
            items[index],
            Line::empty(line_size).fill(Instr::null_value()),
        );

        Instr::reduce(&mut accumulator, item, coordinates, true);
        let plane_dim = CUBE_DIM_X;
        first_index += plane_dim * stride;
        first_coordinate += plane_dim;
    }
    accumulator
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

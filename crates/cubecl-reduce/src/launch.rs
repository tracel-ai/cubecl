use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{LineMode, Reduce, ReduceConfig, ReduceInstruction, ReduceStrategy};

/// Entry point for reduce.
pub fn launch_reduce<R: Runtime, In: Numeric, Out: Numeric, Inst: ReduceInstruction<In>>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    axis: u32,
    cube_count: CubeCount,
    cube_dim: CubeDim,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    match (strategy.use_planes, strategy.shared, config.line_mode) {
        (false, false, LineMode::Contiguous) => unsafe {
            kernel_reduce_contiguous::launch_unchecked::<In, Out, Inst, R>(
                client,
                cube_count,
                cube_dim,
                input.as_tensor_arg(config.line_size as u8),
                output.as_tensor_arg(1),
                ScalarArg::new(axis),
                config.line_size,
                config.bound_checks,
            )
        },
        (false, false, LineMode::Parallel) => unsafe {
            kernel_reduce_parallel::launch_unchecked::<In, Out, Inst, R>(
                client,
                cube_count,
                cube_dim,
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
fn kernel_reduce_contiguous<In: Numeric, Out: Numeric, Inst: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    if bound_checks {
        if ABSOLUTE_POS >= output.len() {
            return;
        }
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
        LineMode::Contiguous,
        false,
    );
    output[ABSOLUTE_POS] = Inst::merge_line::<Out>(out, input.shape(axis_reduce))
}

#[cube(launch_unchecked)]
fn kernel_reduce_parallel<In: Numeric, Out: Numeric, Inst: Reduce<In>>(
    input: &Tensor<Line<In>>,
    output: &mut Tensor<Out>,
    axis_reduce: u32,
    #[comptime] line_size: u32,
    #[comptime] bound_checks: bool,
) {
    let num_active_units = output.len() / line_size;

    if bound_checks {
        if ABSOLUTE_POS >= num_active_units {
            return;
        }
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
        LineMode::Parallel,
        false,
    );

    let out = Inst::to_output_parallel(out, input.shape(axis_reduce));

    #[unroll]
    for k in 0..line_size {
        output[line_size * ABSOLUTE_POS + k] = out[k];
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
pub fn reduce_slice<N: Numeric, Instr: Reduce<N>>(
    items: Slice<Line<N>>,
    start: u32,
    end: u32,
    stride: u32,
    #[comptime] line_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] use_planes: bool,
) -> Instr::Accumulator {
    let mut accumulator = Instr::init_accumulator(line_size);

    let mut index = start;
    let mut coordinate = 0;
    while index < end {
        let coordinates = match comptime!(line_mode) {
            LineMode::Contiguous => {
                let mut coordinates = Line::empty(line_size).fill(coordinate * line_size);
                #[unroll]
                for j in 0..line_size {
                    coordinates[j] += j;
                }
                coordinates
            }
            LineMode::Parallel => Line::empty(line_size).fill(coordinate),
        };
        Instr::reduce(&mut accumulator, items[index], coordinates, use_planes);
        index += stride;
        coordinate += 1;
    }
    accumulator
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

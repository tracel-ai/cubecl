use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::args::init_tensors;
use crate::args::ReduceArgs;
use crate::args::TensorArgs;
use crate::instructions::*;
use crate::primitives::*;
use crate::{LineMode, ReduceConfig, ReduceStrategy};

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
pub(crate) fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: Reduce>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
) {
    let settings = ReduceParams {
        shared: strategy.shared.then(|| {
            if strategy.use_planes {
                config.cube_dim.y
            } else {
                config.cube_dim.num_elems()
            }
        }),
        use_planes: strategy.use_planes,
        line_size: config.line_size,
        line_mode: config.line_mode,
        bound_checks: config.bound_checks,
    };
    unsafe {
        reduce_kernel::launch_unchecked::<In, Out, Rd, TensorArgs, Run>(
            client,
            config.cube_count,
            config.cube_dim,
            input.as_tensor_arg(config.line_size as u8),
            output.as_tensor_arg(1),
            ScalarArg::new(axis),
            settings,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceParams {
    pub shared: Option<u32>, // shared if Some(x) where x is the accumulator size.
    pub use_planes: bool,
    pub line_size: u32,
    pub line_mode: LineMode,
    pub bound_checks: bool,
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, R: Reduce, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    let reduce_index = get_reduce_index(params);

    if params.bound_checks && reduce_index >= get_reduce_count(output.len(), params) {
        terminate!();
    }

    let range = ReduceRange::new::<In, Out>(
        reduce_index,
        &input,
        &mut output,
        axis_reduce,
        params.line_size,
        params.line_mode,
    );

    let accumulator = match comptime!((params.shared, params.use_planes)) {
        (Some(accumulator_size), use_planes) => {
            let mut accumulator = reduce_slice_shared::<In, R::Instruction<In>>(
                &input,
                range,
                accumulator_size,
                params.line_size,
                params.line_mode,
                use_planes,
            );
            sync_units();
            reduce_tree::<In, R::Instruction<In>>(&mut accumulator, accumulator_size)
        }
        (None, true) => reduce_slice_plane::<In, R::Instruction<In>>(
            &input,
            range,
            params.line_size,
            params.line_mode,
        ),
        (None, false) => reduce_slice::<In, R::Instruction<In>>(
            &input,
            range,
            params.line_size,
            params.line_mode,
        ),
    };

    if elected_writer(params) {
        write_to_output::<In, Out, R::Instruction<In>>(
            &mut output,
            accumulator,
            reduce_index,
            input.shape(axis_reduce),
            params,
        );
    }
}

#[cube]
fn get_reduce_index(#[comptime] params: ReduceParams) -> u32 {
    if params.shared.is_some() {
        CUBE_POS
    } else if params.use_planes {
        CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y
    } else {
        ABSOLUTE_POS
    }
}

#[cube]
fn get_reduce_count(output_size: u32, #[comptime] params: ReduceParams) -> u32 {
    match comptime!(params.line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / params.line_size,
    }
}

#[cube]
fn elected_writer(#[comptime] settings: ReduceParams) -> bool {
    if settings.shared.is_some() {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true
    }
}

#[cube]
fn write_to_output<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(accumulator, shape_axis_reduce);

            #[unroll]
            for k in 0..settings.line_size {
                let result: Out = out[k];
                let index = settings.line_size * reduce_index + k;
                output.write(index, Line::cast_from(result));
            }
        }
    }
}

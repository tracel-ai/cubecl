use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;

use crate::BoundChecksInner;
use crate::args::ReduceArgs;
use crate::args::TensorArgs;
use crate::args::init_tensors;
use crate::instructions::*;
use crate::primitives::*;
use crate::{LineMode, ReduceConfig, ReduceStrategy};

/// Launch a reduce kernel. This function assumes that all parameters are already validated.
/// See the main entrypoint `reduce` in `lib.rs` for an example how to call this function
/// with the appropriate assumptions.
pub(crate) fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: ReduceFamily>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    input: TensorHandleRef<Run>,
    output: TensorHandleRef<Run>,
    axis: u32,
    config: ReduceConfig,
    strategy: ReduceStrategy,
    inst: Rd::Config,
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
        line_size_input: config.line_size_input,
        line_size_output: config.line_size_output,
        line_mode: config.line_mode,
        bound_checks: config.bound_checks,
        bound_checks_inner: config.bound_checks_inner,
    };
    unsafe {
        reduce_kernel::launch_unchecked::<In, Out, Rd, TensorArgs, Run>(
            client,
            config.cube_count,
            config.cube_dim,
            input.as_tensor_arg(config.line_size_input as u8),
            output.as_tensor_arg(config.line_size_output as u8),
            ScalarArg::new(axis),
            settings,
            inst,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReduceParams {
    pub shared: Option<u32>, // shared if Some(x) where x is the accumulator size.
    pub use_planes: bool,
    pub line_size_input: u32,
    pub line_size_output: u32,
    pub line_mode: LineMode,
    pub bound_checks: bool,
    pub bound_checks_inner: BoundChecksInner,
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, R: ReduceFamily, RA: ReduceArgs>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: R::Config,
) {
    let (input, mut output) = init_tensors::<RA, In, Out>(input, output);
    let reduce_index = get_reduce_index(params);

    if comptime![params.bound_checks]
        && reduce_index >= get_reduce_count(output.len() * params.line_size_output, params)
    {
        terminate!();
    }

    let range = ReduceRange::new::<In, Out>(reduce_index, &input, &mut output, axis_reduce, params);

    let inst = &R::Instruction::<In>::from_config(config);
    let accumulator = match comptime!((params.shared, params.use_planes)) {
        (Some(accumulator_size), use_planes) => {
            let mut accumulator = reduce_slice_shared::<In, VirtualTensor<In>, R::Instruction<In>>(
                &input,
                inst,
                range,
                accumulator_size,
                params.line_size_input,
                params.line_mode,
                use_planes,
                params.bound_checks_inner,
            );
            sync_units();
            reduce_tree::<In, R::Instruction<In>>(inst, &mut accumulator, accumulator_size)
        }
        (None, true) => reduce_slice_plane::<In, VirtualTensor<In>, R::Instruction<In>>(
            &input,
            inst,
            range,
            params.line_size_input,
            params.line_mode,
            params.bound_checks_inner,
        ),
        (None, false) => reduce_slice::<In, VirtualTensor<In>, R::Instruction<In>>(
            &input,
            range,
            inst,
            params.line_size_input,
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
            inst,
        );
    }
}

#[cube]
pub(crate) fn get_reduce_index(#[comptime] params: ReduceParams) -> u32 {
    if params.shared.is_some() {
        CUBE_POS
    } else if params.use_planes {
        CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y
    } else {
        ABSOLUTE_POS
    }
}

#[cube]
pub(crate) fn get_reduce_count(output_size: u32, #[comptime] params: ReduceParams) -> u32 {
    match comptime!(params.line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / params.line_size_input,
    }
}

#[cube]
pub(crate) fn elected_writer(#[comptime] settings: ReduceParams) -> bool {
    if settings.shared.is_some() {
        UNIT_POS == 0
    } else if settings.use_planes {
        UNIT_POS_X == 0
    } else {
        true.runtime()
    }
}

#[cube]
fn write_to_output<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
    inst: &R,
) {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);

            if comptime![settings.line_size_output == settings.line_size_input] {
                output.write(reduce_index, out);
            } else {
                let num_iters = comptime![settings.line_size_input / settings.line_size_output];

                #[unroll]
                for i in 0..num_iters {
                    let mut tmp = Line::empty(settings.line_size_output);

                    #[unroll]
                    for j in 0..settings.line_size_output {
                        tmp[j] = out[i * settings.line_size_output + j];
                    }

                    let index = num_iters * reduce_index + i;
                    output.write(index, tmp);
                }
            }
        }
    }
}

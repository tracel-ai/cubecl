use crate::{
    LineMode, ReduceFamily, ReduceInstruction,
    args::{ReduceArgs, init_tensors},
    elected_writer, get_reduce_count, get_reduce_index,
    primitives::ReduceRange,
};

pub use super::launch::ReduceParams;

use crate::primitives::*;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

#[cube(launch_unchecked)]
pub fn reduce_kernel_multi<In: Numeric, Out: Numeric, R: ReduceFamily, RA: ReduceArgs>(
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
    let reduced = reduce_kernel_step::<In, Out, R, RA>(&input, range, axis_reduce, params, config);
}

#[cube]
pub fn reduce_kernel_write<Out: Numeric, R: ReduceFamily, RA: ReduceArgs>(
    value: Line<Out>,
    output: &mut VirtualTensor<Out>,
    range: ReduceRange,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: R::Config,
) {
    match comptime!((params.shared, params.use_planes)) {
        (Some(accumulator_size), use_planes) => {
            // use the cude to write to the output.
        }
        (None, true) => {
            // Use the plane to write to the output.
        }
        (None, false) => {
            // Use the unit to write to the output.
        }
    };
}

#[cube]
pub fn reduce_kernel_step<In: Numeric, Out: Numeric, R: ReduceFamily, RA: ReduceArgs>(
    input: &VirtualTensor<In>,
    range: ReduceRange,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: R::Config,
) -> Line<Out> {
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

    let mut out_scalar = Line::empty(params.line_size_output);

    if elected_writer(params) {
        out_scalar = write_to_output::<In, Out, R::Instruction<In>>(
            accumulator,
            input.shape(axis_reduce),
            params,
            inst,
        );
    }

    sync_units();

    out_scalar
}

#[cube]
fn write_to_output<In: Numeric, Out: Numeric, R: ReduceInstruction<In>>(
    accumulator: R::AccumulatorItem,
    shape_axis_reduce: u32,
    #[comptime] settings: ReduceParams,
    inst: &R,
) -> Line<Out> {
    match comptime!(settings.line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            Line::cast_from(result)
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);

            if comptime![settings.line_size_output == settings.line_size_input] {
                out
            } else {
                panic!("Unsupported")
            }
        }
    }
}

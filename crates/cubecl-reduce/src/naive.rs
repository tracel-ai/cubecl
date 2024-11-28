use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{ReduceArgMax, ReduceArgMin, ReduceMean, ReduceProd, ReduceSum};

/// An instruction for the [reduce_naive](reduce_naive) algorithm.
#[cube]
pub trait ReduceNaiveInstruction<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator.
    /// The implement works on lines. Most likely, the accumulator is `Line<T>`
    /// for some CubePrimitive type `T` instead of simply `T`.
    type Accumulator: CubeType;

    /// Initialize the accumulator with a null value for the reduction.
    ///
    /// This could be called many time during reduction. It is required
    /// that reducing the initial accumulator any number of times do not change the outcome
    /// of the reduction. For example, adding 0s in a sum do not change the outcome.
    fn init_accumulator(line_size: u32) -> Self::Accumulator;

    /// Reduce `current_value` into `accumulator`.
    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, i: u32);

    /// Write the result of the reduction stored in `accumulator` into `output[index]`.
    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    );
}

/// A naive implementation of the reduction algorithm.
///
/// Each thread with absolute position P is responsible
/// to compute the reduction corresponding to index P of the `output`.
#[cube]
pub fn reduce_naive<RD: ReduceNaiveInstruction<EI>, EI: Numeric, EO: Numeric>(
    input: &Tensor<Line<EI>>,
    output: &mut Tensor<Line<EO>>,
    dim: u32,
) {
    if ABSOLUTE_POS >= output.len() * output.line_size() {
        return;
    }

    // Compute the first index where to start the reduction for the current thread.
    // First, compute the coordinate corresponding to the ABSOLUTE_POS element of the output tensor
    // Then, use the strides of the input tensor to find the index of the same coordinate
    // in the input tensor.
    let mut offset_input = 0;
    for axis in 0..input.rank() {
        let coordinate = output.coordinate(ABSOLUTE_POS, axis);
        offset_input += coordinate * input.stride(axis);
    }

    // Reduce all the lines along `dim` for the previously computed offset.
    let mut accumulator = RD::init_accumulator(input.line_size());
    for i in 0..input.shape(dim) {
        let index = i * input.stride(dim) + offset_input;
        RD::accumulate(
            &mut accumulator,
            unsafe { *input.index_unchecked(index) },
            i,
        );
    }

    // Write the local outcome into output.
    RD::write::<EO>(output, accumulator, ABSOLUTE_POS, input.shape(dim));
}

// Implementations for common instructions.

#[cube]
impl<EI: Numeric> ReduceNaiveInstruction<EI> for ReduceSum {
    type Accumulator = Line<EI>;

    fn init_accumulator(line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, _i: u32) {
        *accumulator += current_value;
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator);
    }
}

#[cube]
impl<EI: Numeric> ReduceNaiveInstruction<EI> for ReduceProd {
    type Accumulator = Line<EI>;

    fn init_accumulator(line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(1))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, _i: u32) {
        *accumulator *= current_value;
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(accumulator);
    }
}

#[cube]
impl<EI: Numeric> ReduceNaiveInstruction<EI> for ReduceMean {
    type Accumulator = Line<EI>;

    fn init_accumulator(line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, _i: u32) {
        *accumulator += current_value;
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        shape_reduce_dim: u32,
    ) {
        output[index] = Line::cast_from(
            accumulator / Line::empty(output.line_size()).fill(EI::cast_from(shape_reduce_dim)),
        );
    }
}

#[cube]
impl<EI: Numeric> ReduceNaiveInstruction<EI> for ReduceArgMax {
    type Accumulator = (Line<EI>, Line<u32>);

    fn init_accumulator(line_size: u32) -> Self::Accumulator {
        (
            // TODO: switch to using f32::NEG_INFINITY when it's supported: https://github.com/tracel-ai/cubecl/issues/68
            Line::empty(line_size).fill(EI::MIN),
            Line::empty(line_size).fill(0u32),
        )
    }

    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, i: u32) {
        let (max, index) = accumulator;
        #[allow(clippy::collapsible_else_if)]
        if comptime!(current_value.size() > 1) {
            #[unroll]
            for k in 0..current_value.size() {
                if current_value[k] > max[k] {
                    max[k] = current_value[k];
                    index[k] = i;
                }
            }
        } else {
            if current_value > *max {
                *max = current_value;
                *index = Line::new(i);
            }
        }
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        let (_, position) = accumulator;
        output[index] = Line::cast_from(position)
    }
}

#[cube]
impl<EI: Numeric> ReduceNaiveInstruction<EI> for ReduceArgMin {
    type Accumulator = (Line<EI>, Line<u32>);

    fn init_accumulator(line_size: u32) -> Self::Accumulator {
        (
            // TODO: switch to using f32::INFINITY when it's supported: https://github.com/tracel-ai/cubecl/issues/68
            Line::empty(line_size).fill(EI::MAX),
            Line::empty(line_size).fill(0u32),
        )
    }

    fn accumulate(accumulator: &mut Self::Accumulator, current_value: Line<EI>, i: u32) {
        let (min, index) = accumulator;
        #[allow(clippy::collapsible_else_if)]
        if comptime!(current_value.size() > 1) {
            #[unroll]
            for k in 0..current_value.size() {
                if current_value[k] < min[k] {
                    min[k] = current_value[k];
                    index[k] = i;
                }
            }
        } else {
            if current_value < *min {
                *min = current_value;
                *index = Line::new(i);
            }
        }
    }

    fn write<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        index: u32,
        _shape_reduce_dim: u32,
    ) {
        let (_, position) = accumulator;
        output[index] = Line::cast_from(position)
    }
}

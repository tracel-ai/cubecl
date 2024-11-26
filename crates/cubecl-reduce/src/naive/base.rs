use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Specifies the reduce dim algorithm in use
#[cube]
pub trait ReduceDimNaive<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: CubeType;

    /// Initialization for naive algorithm
    fn initialize_naive() -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(accumulator: &mut Self::Accumulator, current_value: EI, i: u32);

    /// Assignation for naive algorithm
    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: u32,
    );
}

#[cube]
pub fn reduce_dim_naive<RD: ReduceDimNaive<EI>, EI: Numeric, EO: Numeric>(
    input: &Tensor<EI>,
    output: &mut Tensor<EO>,
    dim: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    };

    let mut offset_input = 0;

    for i in 0..input.rank() {
        let mut offset_local = ABSOLUTE_POS / output.stride(i);
        offset_local %= output.shape(i);
        if i != dim {
            offset_input += offset_local * input.stride(i);
        }
    }

    let mut accumulator = RD::initialize_naive();

    for i in 0..input.shape(dim) {
        let index = i * input.stride(dim) + offset_input;
        RD::inner_loop_naive(
            &mut accumulator,
            unsafe { *input.index_unchecked(index) },
            i,
        );
    }

    RD::assign_naive::<EO>(output, accumulator, input.shape(dim));
}

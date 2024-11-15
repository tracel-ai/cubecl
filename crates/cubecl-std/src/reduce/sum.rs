use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReduceConfig {
    pub line_size: u32,
    pub plane_size: u32,
    pub num_planes: u32,
}

/// Compute the sum of all elements of `input` and write it to the first element of `output`.
///
/// This is a work in progress toward a more general multi-dimensional reduce kernel.
#[cube(launch_unchecked)]
pub fn reduce_sum<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    #[comptime] config: ReduceConfig,
) {
    reduce_sum_vector(&input.to_slice(), &mut output.to_slice_mut(), config);
}

#[cube]
fn reduce_sum_vector<N: Numeric>(
    input: &Slice<Line<N>>,
    output: &mut SliceMut<Line<N>>,
    #[comptime] config: ReduceConfig,
) {
    let block_size = config.plane_size * config.num_planes;

    // This is an integer division rounded up.
    let num_blocks = input.len() / block_size + (input.len() % block_size > 0) as u32;

    let mut memory = SharedMemory::new_lined(config.plane_size, config.line_size);
    memory[UNIT_POS_X] = Line::empty(config.line_size).fill(N::from_int(0));

    // For each block, we reduce each plane to a single value. Then, we accumulate the results
    // into the memory. Thus, after the loop, the reduction of the first num_planes values
    // of the memory yields the expected output.
    for i in 0..num_blocks {
        let start = i * block_size + UNIT_POS_Y * config.plane_size;
        let sum = plane_sum(input[start + UNIT_POS_X]);
        if UNIT_POS_X == 0 {
            memory[UNIT_POS_Y] += sum;
        }
    }

    // Make sure that each local sum is completed and written to memory.
    sync_units();

    // Sum each elements in memory
    let sum = plane_sum(memory[UNIT_POS_X]);
    if UNIT_POS == 0 {
        output[0] = sum;
    }
}

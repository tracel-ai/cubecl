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
/// This doesn't reduce values accross lines. For a version that does, use [reduce_sum_lined].
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

/// Compute the sum of all elements of `input` and write it to the first element of `output`.
///
/// This reduces values accross lines. For a version that doesn't, use [reduce_sum].
///
/// This is a work in progress toward a more general multi-dimensional reduce kernel.
#[cube(launch_unchecked)]
pub fn reduce_sum_lined<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<N>,
    #[comptime] config: ReduceConfig,
) {
    let mut tmp = SharedMemory::new_lined(1, config.line_size);
    reduce_sum_vector(&input.to_slice(), &mut tmp.to_slice_mut(), config);
    reduce_sum_lines(&tmp.to_slice(), &mut output.to_slice_mut(), 1_u32);
}

/// Compute the sum of all elements of `input` and write it to the first element of `output`.
#[cube]
pub fn reduce_sum_vector<N: Numeric>(
    input: &Slice<Line<N>>,
    output: &mut SliceMut<Line<N>>,
    #[comptime] config: ReduceConfig,
) {
    // How many lines accounted in each iteration.
    let block_size = config.plane_size * config.num_planes;

    // This is an integer division rounded up.
    let num_blocks = input.len() / block_size + (input.len() % block_size > 0) as u32;

    let mut memory = SharedMemory::new_lined(config.num_planes, config.line_size);

    memory[UNIT_POS_Y] = Line::empty(config.line_size).fill(N::from_int(0));

    // For each block, we reduce each group of plane_size lines to a single line. Then, we accumulate the results
    // into the memory. Thus, after the loop, the reduction of the memory yields the expected output.
    for i in 0..num_blocks {
        let index = i * block_size + UNIT_POS_Y * config.plane_size + UNIT_POS_X;
        let value = select(
            index < input.len(),
            input[index],
            Line::empty(config.line_size).fill(N::from_int(0)),
        );
        let sum = plane_sum(value);
        if UNIT_POS_X == 0 {
            memory[UNIT_POS_Y] += sum;
        }
    }

    // Make sure that each local sum is completed and written to memory.
    sync_units();

    // Sum each elements in memory
    let sum = plane_sum(select(
        UNIT_POS_X < config.num_planes,
        memory[UNIT_POS_X],
        Line::empty(config.line_size).fill(N::from_int(0)),
    ));
    if UNIT_POS == 0 {
        output[0] = sum;
    }
}

/// For each line, sum all elements and write the result into the corresponding element of output.
#[cube]
pub fn reduce_sum_lines<N: Numeric>(
    input: &Slice<Line<N>>,
    output: &mut SliceMut<N>,
    #[comptime] length: u32,
) {
    if UNIT_POS < length {
        let line = input[UNIT_POS];

        let mut sum = N::from_int(0);

        #[unroll]
        for k in 0..line.size() {
            sum += line[k];
        }

        output[UNIT_POS] = sum;
    }
}

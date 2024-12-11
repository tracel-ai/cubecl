use cubecl_core::{client::ComputeClient, Runtime};

use crate::matmul::{
    components::{
        stage::*,
        tile::{
            accelerated::{Accelerated16x16x16, Accelerated32x8x16, Accelerated8x32x16},
            plane::{PlaneMma16x16x16, PlaneMma32x8x16, PlaneMma8x32x16},
        },
        InputRuntimeArg, MatmulProblem, MatmulSpec, OutputRuntimeArg,
    },
    kernels::{matmul::base::matmul_cube_preparation, MatmulLaunchError},
};

use super::standard::StandardAlgorithm;

const NUM_SM_APPROX: usize = 50;
const NUM_TENSOR_CORES_APPROX: usize = 8;

pub struct CmmaSelector;

impl CmmaSelector {
    pub fn select_kernel<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: MatmulProblem,
    ) -> Result<(), MatmulLaunchError> {
        let (instruction_m, instruction_n) = find_instruction_shape(problem.m, problem.n);

        let stage_size_m_n = find_stage_size_m_n(
            problem.m,
            problem.n,
            problem.num_batches(),
            NUM_SM_APPROX,
            NUM_TENSOR_CORES_APPROX,
            instruction_m,
            instruction_n,
        );

        match (instruction_m, instruction_n) {
            (16, 16) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, Accelerated16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, Accelerated16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, Accelerated16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, Accelerated16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            (32, 8) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, Accelerated32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, Accelerated32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, Accelerated32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, Accelerated32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            (8, 32) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, Accelerated8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, Accelerated8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, Accelerated8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, Accelerated8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            _ => panic!("No configuration found for instruction shapes."),
        }
    }
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
fn find_instruction_shape(m: usize, n: usize) -> (usize, usize) {
    if m >= 4 * n {
        (32, 8)
    } else if n >= 4 * n {
        (8, 32)
    } else {
        (16, 16)
    }
}

/// A heuristic to find the number of tiles in the stage.
///
/// Maximizes tensor core usage unless doing so would significantly impair
/// parallelization across SMs. It ensures the number of cubes is as close as
/// possible to the available SMs.
fn find_stage_size_m_n(
    m: usize,
    n: usize,
    num_batches: usize,
    num_sm: usize,
    max_tensor_cores: usize,
    instruction_m: usize,
    instruction_n: usize,
) -> usize {
    let mut dim_num_tiles = max_tensor_cores;

    let total_tiles_m = (m + instruction_m - 1) / instruction_m;
    let total_tiles_n = (n + instruction_n - 1) / instruction_n;

    let total_tiles = total_tiles_m * total_tiles_n * num_batches;

    let mut stage_num_tiles = dim_num_tiles * dim_num_tiles;
    let mut num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;

    // We keep track of two configurations to select the closest to `num_sm`, whether it's a bit over or under
    let mut previous_dim_num_tiles = dim_num_tiles;
    let mut previous_num_cubes = num_cubes_expected;

    // Refine tensor core usage to stay as close as possible to `num_sm`
    while num_cubes_expected < num_sm && stage_num_tiles > 1 {
        previous_dim_num_tiles = dim_num_tiles;
        previous_num_cubes = num_cubes_expected;

        // Reduce tensor core usage
        dim_num_tiles = (dim_num_tiles + 1) / 2;
        stage_num_tiles = dim_num_tiles * dim_num_tiles;

        // Number of cubes grows as a consequence of smaller stage
        num_cubes_expected = (total_tiles + stage_num_tiles - 1) / stage_num_tiles;
    }

    // Compare previous and current values to determine the closest to `num_sm`
    if (previous_num_cubes as isize - num_sm as isize).abs()
        <= (num_cubes_expected as isize - num_sm as isize).abs()
    {
        previous_dim_num_tiles
    } else {
        dim_num_tiles
    }
}

pub struct PlaneMmaSelector;

impl PlaneMmaSelector {
    pub fn select_kernel<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        problem: MatmulProblem,
    ) -> Result<(), MatmulLaunchError> {
        let (instruction_m, instruction_n) = find_instruction_shape(problem.m, problem.n);

        let stage_size_m_n = find_stage_size_m_n(
            problem.m,
            problem.n,
            problem.num_batches(),
            NUM_SM_APPROX,
            NUM_TENSOR_CORES_APPROX,
            instruction_m,
            instruction_n,
        );

        match (instruction_m, instruction_n) {
            (16, 16) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, PlaneMma16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, PlaneMma16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, PlaneMma16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, PlaneMma16x16x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            (32, 8) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, PlaneMma32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, PlaneMma32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, PlaneMma32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, PlaneMma32x8x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            (8, 32) => match stage_size_m_n {
                1 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S1x1x2, PlaneMma8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                2 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S2x2x2, PlaneMma8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                4 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S4x4x2, PlaneMma8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                8 => matmul_cube_preparation::<
                    MS,
                    R,
                    StandardAlgorithm<MS, S8x8x2, PlaneMma8x32x16<MS::ES, MS::EA>>,
                >(client, input, output, problem),
                _ => panic!("No configuration found for this stage size. "),
            },
            _ => panic!("No configuration found for instruction shapes."),
        }
    }
}

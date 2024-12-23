use cubecl_core::{client::ComputeClient, ir::Elem, prelude::CubePrimitive, Feature, Runtime};
use cubecl_runtime::DeviceProperties;

use crate::matmul::{
    components::{
        stage::*,
        tile::{accelerated::Accelerated, plane::PlaneMma},
        InputRuntimeArg, MatmulProblem, MatmulSelection, MatmulSize, MatmulSpec, OutputRuntimeArg,
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
        plane_dim: u32,
    ) -> Result<(), MatmulLaunchError> {
        let (instruction_m, instruction_n, instruction_k) = find_instruction_shape(
            Some((
                client.properties(),
                (
                    MS::ES::as_elem_native().unwrap(),
                    MS::ES::as_elem_native().unwrap(),
                    MS::EA::as_elem_native().unwrap(),
                ),
            )),
            problem.m,
            problem.n,
        );

        let stage_size_m_n = find_stage_size_m_n(
            problem.m,
            problem.n,
            problem.num_batches(),
            NUM_SM_APPROX,
            NUM_TENSOR_CORES_APPROX,
            instruction_m,
            instruction_n,
        );

        let selection = MatmulSelection {
            tile: MatmulSize {
                m: instruction_m as u32,
                n: instruction_n as u32,
                k: instruction_k as u32,
            },
            num_stagess: MatmulSize {
                m: stage_size_m_n as u32,
                n: stage_size_m_n as u32,
                k: 2,
            },
            plane_dim,
        };
        let config_input = CommonStageInput {
            tile: selection.tile.clone(),
            num_stages: selection.num_stagess.clone(),
        };

        matmul_cube_preparation::<MS, R, StandardAlgorithm<Accelerated>>(
            client,
            input,
            output,
            problem,
            config_input,
            selection,
        )
    }
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
#[allow(clippy::type_complexity)]
fn find_instruction_shape(
    properties: Option<(&DeviceProperties<Feature>, (Elem, Elem, Elem))>,
    m: usize,
    n: usize,
) -> (usize, usize, usize) {
    let supported = |m: u8, n: u8, k: u8| {
        properties
            .map(|(p, (a, b, c))| p.feature_enabled(Feature::Cmma { a, b, c, m, n, k }))
            .unwrap_or(true)
    };

    if m >= 4 * n && supported(32, 8, 16) {
        (32, 8, 16)
    } else if n >= 4 * n && supported(8, 32, 16) {
        (8, 32, 16)
    } else if supported(16, 16, 16) {
        (16, 16, 16)
    } else {
        (16, 16, 8)
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
        plane_dim: u32,
    ) -> Result<(), MatmulLaunchError> {
        let (instruction_m, instruction_n, instruction_k) =
            find_instruction_shape(None, problem.m, problem.n);

        let stage_size_m_n = find_stage_size_m_n(
            problem.m,
            problem.n,
            problem.num_batches(),
            NUM_SM_APPROX,
            NUM_TENSOR_CORES_APPROX,
            instruction_m,
            instruction_n,
        );

        let selection = MatmulSelection {
            tile: MatmulSize {
                m: instruction_m as u32,
                n: instruction_n as u32,
                k: instruction_k as u32,
            },
            num_stagess: MatmulSize {
                m: stage_size_m_n as u32,
                n: stage_size_m_n as u32,
                k: 2,
            },
            plane_dim,
        };
        let config_input = CommonStageInput {
            tile: selection.tile.clone(),
            num_stages: selection.num_stagess.clone(),
        };

        matmul_cube_preparation::<MS, R, StandardAlgorithm<PlaneMma>>(
            client,
            input,
            output,
            problem,
            config_input,
            selection,
        )
    }
}

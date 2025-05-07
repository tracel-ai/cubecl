use std::cmp::min;

use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::{Feature, Runtime, client::ComputeClient, ir::Elem, prelude::CubePrimitive};
use cubecl_runtime::DeviceProperties;

use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{InputRuntimeArg, OutputRuntimeArg};
use crate::matmul::{
    components::{
        CompleteStageTiling, InputArg, MatmulPrecision, MatmulProblem, MatmulSelection, MatmulSize,
        MatmulSpec, OutputArg,
        global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
        tile::TileMatmulFamily,
    },
    kernels::{MatmulLaunchError, matmul::base::matmul_cube_preparation},
};

use super::Algorithm;
use super::base::MultiRowStrategy;

pub(crate) const NUM_SM_APPROX: u32 = 50;
pub(crate) const NUM_TENSOR_CORES_APPROX: u32 = 4;
const NUM_PLANES_PER_TENSOR_CORES: u32 = 2;

/// Select which kernel to launch for the given Algorithm.
///
/// Only works for concrete tensor inputs and output.
#[allow(clippy::result_large_err)]
pub fn select_kernel_concrete<MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    lhs: &TensorHandleRef<'_, R>,
    rhs: &TensorHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError>
where
    InputArg<MS>: ConcreteInputsFactory,
    OutputArg<MS>: ConcreteOutputFactory,
{
    let selection = matmul_selection::<A::TileMatmul, R>(
        client,
        &problem,
        plane_dim,
        A::multi_row_strategy(),
        <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked(),
        <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked(),
    );
    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape,
        tile_count: selection.tile_count,
    };

    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    matmul_cube_preparation::<MS, R, A>(
        client,
        <InputArg<MS> as ConcreteInputsFactory>::create(lhs, rhs, &selection, &problem),
        <OutputArg<MS> as ConcreteOutputFactory>::create(out, &selection, &problem),
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    )
}

/// Select which kernel to launch for the given Algorithm.
pub fn select_kernel_virtual<'a, MS: MatmulSpec, R: Runtime, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: InputRuntimeArg<'a, MS, R>,
    output: OutputRuntimeArg<'a, MS, R>,
    problem: MatmulProblem,
    plane_dim: u32,
) -> Result<(), MatmulLaunchError> {
    let selection = matmul_selection::<A::TileMatmul, R>(
        client,
        &problem,
        plane_dim,
        A::multi_row_strategy(),
        <MS::Precision as MatmulPrecision>::ES::as_elem_native_unchecked(),
        <MS::Precision as MatmulPrecision>::EA::as_elem_native_unchecked(),
    );
    let config_input = CompleteStageTiling {
        tile_shape: selection.tile_shape,
        tile_count: selection.tile_count,
    };
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    matmul_cube_preparation::<MS, R, A>(
        client,
        input,
        output,
        problem,
        (
            (
                config_input,
                A::stage_buffering_strategy(),
                vectorization,
                A::num_stages(),
            ),
            A::loading_precompute_strategy(),
        ),
        selection,
    )
}

/// A heuristic to choose the instruction to use, based on input shape
///
/// Will use 16x16 for balanced matrices, and 32x8 or 8x32 for degenerated ones.
#[allow(clippy::type_complexity)]
pub(crate) fn find_instruction_shape(
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
    } else if supported(8, 8, 8) {
        (8, 8, 8)
    } else {
        (16, 16, 8)
    }
}

/// A heuristic to find the number of tiles in the stage.
///
/// Maximizes tensor core usage unless doing so would significantly impair
/// parallelization across SMs. It ensures the number of cubes is as close as
/// possible to the available SMs.
pub(crate) fn find_stage_size(
    m: usize,
    n: usize,
    num_batches: usize,
    num_sm: usize,
    virtual_tensor_cores: usize,
    instruction_m: usize,
    instruction_n: usize,
) -> usize {
    let min_inst = instruction_m.min(instruction_n);
    let max_tiles = 256 / min_inst;
    let mut dim_num_tiles = virtual_tensor_cores.min(max_tiles);

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

pub fn matmul_selection<TMM: TileMatmulFamily, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    problem: &MatmulProblem,
    plane_dim: u32,
    multi_row_strategy: MultiRowStrategy,
    elem_stage: Elem,
    elem_acc: Elem,
) -> MatmulSelection {
    let (instruction_m, instruction_n, instruction_k) = find_instruction_shape(
        if TMM::requires_tensor_cores() {
            Some((client.properties(), (elem_stage, elem_stage, elem_acc)))
        } else {
            None
        },
        problem.m,
        problem.n,
    );

    let num_tensor_cores = client
        .properties()
        .hardware_properties()
        .num_tensor_cores
        .unwrap_or(NUM_TENSOR_CORES_APPROX);
    // The number of planes that can send tasks to tensor cores.
    //
    // Going over 8 might use too much shared memory.
    let tensor_cores_channels = min(8, num_tensor_cores * NUM_PLANES_PER_TENSOR_CORES) as usize;

    let stage_size = find_stage_size(
        problem.m,
        problem.n,
        problem.num_batches(),
        client
            .properties()
            .hardware_properties()
            .num_streaming_multiprocessors
            .unwrap_or(NUM_SM_APPROX) as usize,
        tensor_cores_channels,
        instruction_m,
        instruction_n,
    );

    let (rows_per_plane, stage_size_m) =
        change_rows_per_plane(multi_row_strategy, stage_size, instruction_m, problem.m);

    // Makes all rows the length of plane_dim
    let k = plane_dim / instruction_k as u32;

    MatmulSelection {
        tile_shape: MatmulSize {
            m: instruction_m as u32,
            n: instruction_n as u32,
            k: instruction_k as u32,
        },
        tile_count: MatmulSize {
            m: stage_size_m as u32,
            n: stage_size as u32,
            k,
        },
        plane_dim,
        rows_per_plane: rows_per_plane as u32,
    }
}

fn change_rows_per_plane(
    strategy: MultiRowStrategy,
    stage_size: usize,
    instruction_m: usize,
    problem_m: usize,
) -> (usize, usize) {
    let use_multi = match strategy {
        MultiRowStrategy::Never => false,
        MultiRowStrategy::Always => true,
        MultiRowStrategy::Adaptive {
            minimum_stage_count,
        } => problem_m > stage_size * instruction_m * minimum_stage_count,
    };

    let rows = if use_multi { 2 } else { 1 };
    (rows, stage_size * rows)
}

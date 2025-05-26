use crate::matmul::components::MatmulSize;
use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{MatmulProblem, MatrixLayout};
use crate::matmul::kernels::matmul::{Algorithm, PlaneMatmulSelection};
use crate::matmul::tests::cmma_matmul::tma_test_launcher::test_tma_matmul_algorithm;
use crate::matmul::tests::test_utils::TestPrecision;
use cubecl_core::Runtime;

pub fn test_algo<
    A: Algorithm<MatmulSelection = PlaneMatmulSelection>,
    P: TestPrecision,
    R: Runtime,
>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_shape: MatmulSize,
    tile_count: MatmulSize,
    problem: MatmulSize,
) {
    let client = R::client(&Default::default());
    let plane_dim = match client.properties().hardware.defined_plane_size() {
        Some(val) => val,
        None => {
            println!("Can't run test without a fixed plane size.");
            return;
        }
    };

    let problem = MatmulProblem {
        m: problem.m as usize,
        n: problem.n as usize,
        k: problem.k as usize,
        batches: (vec![2], vec![2]),
        lhs_layout: layouts.0,
        rhs_layout: layouts.1,
    };

    let selection = PlaneMatmulSelection {
        tile_shape,
        tile_count,
        plane_dim,
        rows_per_plane: 1,
    };
    let config_input = (&selection).into();

    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };
    test_tma_matmul_algorithm::<A, P, R>(
        client,
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
    );
}

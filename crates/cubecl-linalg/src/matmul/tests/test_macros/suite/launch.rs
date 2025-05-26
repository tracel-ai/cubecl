use crate::matmul::components::MatmulSize;
use crate::matmul::components::stage::StageVectorization;
use crate::matmul::components::{MatmulProblem, MatrixLayout};
use crate::matmul::kernels::matmul::{Algorithm, PlaneMatmulSelection, UnitMatmulSelection};
use crate::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_algorithm;
use crate::matmul::tests::cmma_matmul::tma_test_launcher::test_tma_matmul_algorithm;
use crate::matmul::tests::test_utils::TestPrecision;
use cubecl_core::Runtime;

#[macro_export]
macro_rules! testgen_matmul_launch {
    ($kind: ident, $algorithm: ty, $precision: ty, $layout_lhs: ident, $layout_rhs: ident, $tile: expr, $stage: expr, $problem: expr) => {
        use super::*;

        #[test]
        pub fn test() {
            cubecl_linalg::matmul::tests::test_macros::suite::launch::test_algo::<
                $algorithm,
                $precision,
                TestRuntime,
            >(
                (MatrixLayout::$layout_lhs, MatrixLayout::$layout_rhs),
                $tile,
                $stage,
                $problem,
            );
        }
    };
}

pub fn test_algo<
    A: Algorithm<MatmulSelection = PlaneMatmulSelection>,
    P: TestPrecision,
    R: Runtime,
>(
    layouts: (MatrixLayout, MatrixLayout),
    tile_shape: MatmulSize,
    tile_count: MatmulSize,
    problem: MatmulSize,
    // rows_per_plane: u32,
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

    // TODO merge
    // let rows_per_plane = A::accumulator_shape();

    let selection = PlaneMatmulSelection {
        tile_shape,
        tile_count,
        plane_dim,
        // TODO
        rows_per_plane: 1,
    };
    let config_input = (&selection).into();
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };

    test_matmul_algorithm::<A, P, R>(
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

pub fn test_algo_unit<
    A: Algorithm<MatmulSelection = UnitMatmulSelection>,
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

    let selection = UnitMatmulSelection {
        tile_shape,
        tile_count,
        plane_dim,
    };
    let config_input = (&selection).into();
    let vectorization = StageVectorization {
        stage_line_size: 0,
        stage_elem_padding: 0,
    };

    test_matmul_algorithm::<A, P, R>(
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

pub fn test_algo_tma<
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

use crate::matmul::{
    components::{
        global,
        stage::{self, S1x1x1, S4x4x2},
        tile::plane::PlaneMma16x16x16,
    },
    kernels::cmma_matmul::Algorithm,
};

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_launch {
    ($eg:ty,) => {
        use cubecl_linalg::matmul::tests::cmma_matmul::matmul_test_launcher::test_matmul_launch;
        use cubecl_linalg::tensor::TensorHandle;

        #[test]
        pub fn test_launch_matmul_b3x4_g300x200x250_col_row() {
            type EG = $eg;
            let problem = MatmulProblem::<EG> {
                m: 300,
                n: 200,
                k: 250,
                batches: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
                _element: PhantomData,
            };

            test_matmul_launch::<EG, TestRuntime>(problem, false, &Default::default());
        }
    };
}

mod asdf {
    use std::marker::PhantomData;

    use cubecl_core::{client, CubeCount, CubeDim, Runtime};

    use crate::matmul::{
        base,
        components::{
            batch, global,
            stage::{self, S4x4x2},
            tile::plane::PlaneMma16x16x16,
            MatmulProblem, MatrixLayout,
        },
        kernels::cmma_matmul::{self, AdvancedConfig},
        tests::cmma_matmul::matmul_test_launcher::test_matmul_internal,
    };

    pub fn test_batch_matmul_b3x4_g300x300x300_s4x4x2<R: Runtime>(device: &R::Device) {
        const plane_dim: u32 = 32;
        type eg = f32;
        type es = f32;
        type ea = f32;
        type i_16x16x16<I, O> = PlaneMma16x16x16<I, O>;

        let problem = MatmulProblem {
            m: 300,
            n: 300,
            k: 300,
            batches: vec![3, 4],
            lhs_layout: MatrixLayout::ColMajor,
            rhs_layout: MatrixLayout::ColMajor,
            lhs_line_size: 4,
            rhs_line_size: 4,
            out_line_size: 4,
        };

        struct Test {}
        impl cmma_matmul::Algorithm<eg> for Test {
            const PLANE_DIM: u32 = plane_dim;
            type EG = eg;
            type ES = es;
            type EA = ea;

            type TileMatmul = i_16x16x16<Self::ES, Self::EA>;

            type StageSize = S4x4x2;

            type StageMatmul = stage::row_accumulate::Matmul<
                Self::ES,
                Self::EG,
                Self::EA,
                Self::TileMatmul,
                Self::StageSize,
            >;

            type LhsLoader = global::tensor_view::LhsLoader<Self::EG, Self::ES>;
            type RhsLoader = global::tensor_view::RhsLoader<Self::EG, Self::ES>;
            type Unloader = global::tensor_view::Unloader<Self::EG>;
            type GlobalMatmul = global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;

            type BatchMatmul = batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

            fn cube_dim() -> CubeDim {
                CubeDim::new(32, 4, 1)
            }

            fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                CubeCount::Static(5, 5, 12)
            }
        }
        let advanced_config = AdvancedConfig::default();

        test_matmul_internal::<Test, eg, R>(problem, advanced_config, device);
    }
}

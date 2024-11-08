// Tests nomenclature:
// batch: [oo=one_to_one, om=one_to_many]b[batch dims]
// global: g[m]x[n]x[k], with m,n,k the whole matrix dimensions
// stage: s[m]x[n]x[k], with m,n,k the number of tiles along those dims
// tile: t[m]x[n]x[k], with m,n,k the tile dimensions. tile algorithm is given by macro arguments
// layouts: [r/c][r/c], r=row, c=col, respectively for lhs and rhs
// line size: ln[v], with v the line size of all tensors. if different then ln[v_lhs]x[v_rhs]x[v_out]

#[allow(missing_docs)]
#[macro_export]
macro_rules! matmul_test_define {
    (
        $t_16x16x16:ident,
        $t_32x8x16:ident,
        $t_8x32x16:ident,
        $eg:ty,
        $es:ty,
        $ea:ty,
        $plane_dim:expr
    ) => {
        #[test]
        pub fn oob3x4_g300x300x300_s4x4x2_t16x16x16_cc_ln4() {
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
            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S4x4x2;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::row_accumulate::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new(32, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(5, 5, 12)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_internal::<Test, $eg, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }
    };
}

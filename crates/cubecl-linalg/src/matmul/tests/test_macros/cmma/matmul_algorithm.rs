// Tests nomenclature:
// batch: b[o=one_to_one, m=one_to_many][batch dims, optional]
// global: g[h=homogeneous, pc=producer_consumer][m]x[n]x[k], with m,n,k the whole matrix dimensions
// stage: s[m]x[n]x[k], with m,n,k the number of tiles along those dims
// tile: t[m]x[n]x[k], with m,n,k the tile dimensions. tile algorithm is given by macro arguments
// layouts: [r/c][r/c], r=row, c=col, respectively for lhs and rhs
// line size: ln[v], with v the line size of all tensors. if different then ln[v_lhs]x[v_rhs]x[v_out]
// Other specifications may be appended at the end

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
        pub fn bo1_gpc16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::single_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::producer_consumer::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gpc32x16x16_s2x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::single_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::producer_consumer::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gpc16x16x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gpc16x32x16_s1x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x2x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::single_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::producer_consumer::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gpc64x64x64_s2x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::single_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::producer_consumer::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 2)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gpc16x16x32_s1x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x2;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::single_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::producer_consumer::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Self::EG, Self::ES, Self::GlobalMatmul>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gh16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gh32x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gh16x32x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gh16x16x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm6_gh16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![2, 3],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gh32x32x32_s1x1x1_t16x16x16_rr_ln4_colspan() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: vec![2],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::ColMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gh32x32x32_s1x1x1_t16x16x16_rr_ln4_swizzlespan() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: vec![2],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::SwizzleSpanMatmul<2>,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 2)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm5_gh16x16x16_s1x1x1_t16x16x16_rr_ln4_cubez2() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![5],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;

                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
                    Self::ES,
                    Self::EG,
                    Self::EA,
                    Self::TileMatmul,
                    Self::StageSize,
                >;
                type GlobalMatmul =
                    global::homogeneous::Matmul<Self::EG, Self::ES, Self::StageMatmul>;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Self::EG,
                    Self::ES,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 2)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gh300x300x300_s4x4x2_t16x16x16_cc_ln4() {
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(5, 5, 12)
                }
            }

            let advanced_config = AdvancedConfig::default();

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gh108x108x243_s4x4x2_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 108,
                n: 108,
                k: 243,
                batches: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 12)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gh256x256x256_s4x4x2_t16x16x16_cr_ln2x2x4() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 12)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gh256x256x256_s4x4x2_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: vec![3],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 3)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gh16x16x16_s1x1x1_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![3],
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 3)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gh16x16x16_s1x1x1_t16x16x16_R() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![3],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 3)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh256x256x256_s4x4x2_t16x16x16_rc_ln4_ymajor() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                tiling_order: TilingOrderConfig::YMajor,
                ..Default::default()
            };

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s1x1x1_t16x16x16_cc_ln4_ymajor() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                tiling_order: TilingOrderConfig::YMajor,
                ..Default::default()
            };

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s1x1x1_t16x16x16_R() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x14x16_s1x1x1_t16x16x16_rc_ln4x4x2() {
            let problem = MatmulProblem {
                m: 16,
                n: 14,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 2,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x12x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 12,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x12_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 12,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh60x60x120_s4x4x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 60,
                n: 60,
                k: 120,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x36_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 36,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh12x12x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 12,
                n: 12,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rr_ln1() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                ..Default::default()
            };

            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_cr_ln4_lhs_row_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::RowMajor), None),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_cr_ln4_lhs_row_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::RowMajor), None),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh256x256x256_s4x4x2_t16x16x16_cr_ln2x2x4_lhs_row_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: vec![3, 4],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 12)
                }
            }

            let advanced_config = AdvancedConfig {
                enforced_tile_layout: (Some(MatrixLayout::RowMajor), Some(MatrixLayout::ColMajor)),
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_rc_ln1() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh8x32x16_s1x1x1_t8x32x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: vec![],
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_8x32x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_rr_ln2() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
                out_line_size: 2,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s2x2x2_t16x16x16_rr_ln4_ymajor() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig {
                tiling_order: TilingOrderConfig::YMajor,
                ..Default::default()
            };
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x16_s1x1x1_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: vec![],
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x16x128_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 128,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x16x128_s2x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 128,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x224_s2x2x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 224,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh16x32x16_s1x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S1x2x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s2x2x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x16_s2x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s2x2x2_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s2x2x2_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x32x32_s2x2x2_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: vec![],
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
                type StageSize = S2x2x2;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x16x16_s2x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S2x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh32x8x16_s1x1x1_t32x8x16_cc_ln1() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S1x1x1;
                type TileMatmul = $t_32x8x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh128x16x16_s8x1x1_t16x16x16_rr_ln1() {
            let problem = MatmulProblem {
                m: 128,
                n: 16,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<$eg> for Test {
                const PLANE_DIM: u32 = $plane_dim;
                type EG = $eg;
                type ES = $es;
                type EA = $ea;
                type StageSize = S8x1x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 8, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh64x64x16_s4x4x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 16,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageSize = S4x4x1;
                type TileMatmul = $t_16x16x16<Self::ES, Self::EA>;
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gh64x64x32_s4x4x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 32,
                batches: vec![],
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
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
                type StageMatmul = stage::multi_buffer::Matmul<
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
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            let advanced_config = AdvancedConfig::default();
            test_matmul_algorithm::<Test, $eg, $es, TestRuntime>(
                problem,
                advanced_config,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }
    };
}

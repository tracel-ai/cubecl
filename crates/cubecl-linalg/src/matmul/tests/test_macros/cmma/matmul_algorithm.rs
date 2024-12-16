// Tests nomenclature:
// batch: b[o=one_to_one, m=one_to_many][batch dims, optional]
// global: g[fl=full_load, bp=buffer/pipelined, bs=buffer/specialized][m]x[n]x[k], with m,n,k the whole matrix dimensions
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
        $plane_dim:expr
    ) => {
        #[test]
        pub fn bo4_gbp256x256x256_s4x4x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: (vec![4], vec![4]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::buffered::pipelined::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 4)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbp16x16x256_s1x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 256,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x2>;
                type GlobalMatmul = global::buffered::pipelined::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbp16x16x32_s1x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x2>;
                type GlobalMatmul = global::buffered::pipelined::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs128x256x256_s4x4x2_t16x16x16_cc_ln4_transposed_cube_count() {
            let problem = MatmulProblem {
                m: 128,
                n: 256,
                k: 256,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::TransposedDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(problem: &MatmulProblem) -> CubeCount {
                    let m_stage = S4x4x2::NUM_M * 16;
                    let n_stage = S4x4x2::NUM_N * 16;
                    let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
                    let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

                    use cubecl_linalg::matmul::components::batch::CubeCountDispatch;
                    batch::TransposedDispatch::cube_count(cubes_needed_m, cubes_needed_n, 1u32)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo4_4x3_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![1, 4], vec![4, 3]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 16)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs16x16x480_s1x1x3_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 480,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x3>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gfl1000x16x16_s1x1x1_t16x16x16_rr_ln4_transposed_dispatch() {
            let problem = MatmulProblem {
                m: 1024,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::TransposedDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 64, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gbs300x300x300_s4x4x2_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 300,
                n: 300,
                k: 300,
                batches: (vec![3, 4], vec![3, 4]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 8, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(5, 5, 12)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs16x32x32_s1x2x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x2x2>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs32x16x32_s2x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x1x2>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 3, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs16x16x128_s1x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 128,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x2>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo1_gbs16x16x32_s1x1x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::single_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x2>;
                type GlobalMatmul = global::buffered::specialized::Matmul<Spec, Self::StageMatmul>;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::RowMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gfl32x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gfl16x32x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm1_gfl16x16x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm6_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![2, 3], vec![2, 3]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gfl32x32x32_s1x1x1_t16x16x16_rr_ln4_colspan() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: (vec![2], vec![2]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::ColMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gfl32x32x32_s1x1x1_t16x16x16_rr_ln4_swizzlespan() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: (vec![2], vec![2]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::SwizzleSpanMatmul<2>,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 2)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gfl32x32x32_s1x1x1_t16x16x16_rr_ln4_transposed_dispatch() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: (vec![2], vec![2]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::SwizzleSpanMatmul<2>,
                    batch::TransposedDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 2)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gfl160x256x16_s1x1x1_t16x16x16_rr_ln4_swizzle_x_dispatch() {
            let problem = MatmulProblem {
                m: 160,
                n: 256,
                k: 16,
                batches: (vec![2], vec![2]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::SwizzleSpanMatmul<2>,
                    batch::SwizzleNaturalDispatch<2>,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(10, 16, 2)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm2_gfl160x256x16_s1x1x1_t16x16x16_rr_ln4_swizzle_y_dispatch() {
            let problem = MatmulProblem {
                m: 160,
                n: 256,
                k: 16,
                batches: (vec![2], vec![2]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::SwizzleSpanMatmul<2>,
                    batch::SwizzleTransposedDispatch<2>,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(16, 10, 2)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bm5_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4_cubez2() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![5], vec![5]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul = batch::one_to_many::Matmul<
                    Spec,
                    Self::GlobalMatmul,
                    batch::RowMajorSpanMatmul,
                    batch::NaturalDispatch,
                >;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }

                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 2)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gfl300x300x300_s4x4x2_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 300,
                n: 300,
                k: 300,
                batches: (vec![3, 4], vec![3, 4]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };

            struct Test {}
            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(5, 5, 12)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gfl108x108x243_s4x4x2_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 108,
                n: 108,
                k: 243,
                batches: (vec![3, 4], vec![3, 4]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 12)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gfl256x256x256_s4x4x2_t16x16x16_cr_ln2x2x4() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: (vec![3, 4], vec![3, 4]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 12)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gfl256x256x256_s4x4x2_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: (vec![3], vec![3]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 3)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gfl16x16x16_s1x1x1_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![3], vec![3]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 3)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3_gfl16x16x16_s1x1x1_t16x16x16_R() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![3], vec![3]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 3)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl256x256x256_s4x4x2_t16x16x16_rc_ln4_col_major() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::ColMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s1x1x1_t16x16x16_cc_ln4_col_major() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::ColMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(2, 2, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x14x16_s1x1x1_t16x16x16_rc_ln4x4x2() {
            let problem = MatmulProblem {
                m: 16,
                n: 14,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 2,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x12x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 12,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x12_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 12,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl60x60x120_s4x4x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 60,
                n: 60,
                k: 120,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x36_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 36,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl12x12x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 12,
                n: 12,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rr_ln1() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rc_ln1() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_cc_ln1() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_cr_ln1() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_cr_ln4_lhs_row_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (Some(MatrixLayout::RowMajor), None),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_rr_ln4_lhs_col_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (Some(MatrixLayout::ColMajor), None),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_cr_ln4_lhs_row_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (Some(MatrixLayout::RowMajor), None),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_rr_ln4_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::ColMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_rc_ln4_rhs_row_enforced() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (None, Some(MatrixLayout::RowMajor)),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo3x4_gfl256x256x256_s4x4x2_t16x16x16_cr_ln2x2x4_lhs_row_rhs_col_enforced() {
            let problem = MatmulProblem {
                m: 256,
                n: 256,
                k: 256,
                batches: (vec![3, 4], vec![3, 4]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(4, 4, 12)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        enforced_tile_layout: (
                            Some(MatrixLayout::RowMajor),
                            Some(MatrixLayout::ColMajor),
                        ),
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_rc_ln1() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl8x32x16_s1x1x1_t8x32x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 8,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_8x32x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_rr_ln2() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 2,
                rhs_line_size: 2,
                out_line_size: 2,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s2x2x2_t16x16x16_rr_ln4_col_major() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }

                fn advanced_config() -> AdvancedConfig {
                    AdvancedConfig {
                        lhs_tiling_order: TilingOrderConfig::ColMajor,
                        rhs_tiling_order: TilingOrderConfig::ColMajor,
                        ..Default::default()
                    }
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x32_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x16_s1x1x1_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x16x128_s1x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 16,
                k: 128,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x16x128_s2x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 128,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x224_s2x2x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 224,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl16x32x16_s1x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 16,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x2x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s2x2x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x16_s2x2x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s2x2x2_t16x16x16_rc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s2x2x2_t16x16x16_cr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x32x32_s2x2x2_t16x16x16_cc_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 32,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x2x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x16x16_s2x1x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x8x16_s1x1x1_t32x8x16_cc_ln1() {
            let problem = MatmulProblem {
                m: 32,
                n: 8,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::ColMajor,
                rhs_layout: MatrixLayout::ColMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_32x8x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S1x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 1, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl128x16x16_s8x1x1_t16x16x16_rr_ln1() {
            let problem = MatmulProblem {
                m: 128,
                n: 16,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 1,
                rhs_line_size: 1,
                out_line_size: 1,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S8x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 8, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl64x64x16_s4x4x1_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 16,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl64x64x32_s4x4x2_t16x16x16_rr_ln4() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x16x32_s2x1x2_t16x16x16_rr_ln4_tilewise_load_lhs() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x1x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::TilewiseLoading,
                    global::full_load::CyclicLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl32x16x32_s2x1x2_t16x16x16_rr_ln4_tilewise_load_rhs() {
            let problem = MatmulProblem {
                m: 32,
                n: 16,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S2x1x2>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::CyclicLoading,
                    global::full_load::TilewiseLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 2, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }

        #[test]
        pub fn bo_gfl64x64x32_s4x4x1_t16x16x16_rr_ln4_tilewise_load_both() {
            let problem = MatmulProblem {
                m: 64,
                n: 64,
                k: 32,
                batches: (vec![], vec![]),
                lhs_layout: MatrixLayout::RowMajor,
                rhs_layout: MatrixLayout::RowMajor,
                lhs_line_size: 4,
                rhs_line_size: 4,
                out_line_size: 4,
            };
            struct Test {}

            impl matmul::Algorithm<Spec> for Test {
                const PLANE_DIM: u32 = $plane_dim;

                type TileMatmul = $t_16x16x16<ES, EA>;
                type StageMatmul =
                    stage::multi_buffer::Matmul<ES, EG, EA, Self::TileMatmul, S4x4x1>;
                type GlobalMatmul = global::full_load::Matmul<
                    Spec,
                    Self::StageMatmul,
                    global::full_load::TilewiseLoading,
                    global::full_load::TilewiseLoading,
                >;
                type BatchMatmul =
                    batch::one_to_one::Matmul<Spec, Self::GlobalMatmul, batch::NaturalDispatch>;

                fn cube_dim() -> CubeDim {
                    CubeDim::new($plane_dim, 4, 1)
                }
                fn cube_count(_problem: &MatmulProblem) -> CubeCount {
                    CubeCount::Static(1, 1, 1)
                }
            }

            test_matmul_algorithm::<Test, EG, ES, TestRuntime>(
                problem,
                &<<TestRuntime as Runtime>::Device>::default(),
            );
        }
    };
}

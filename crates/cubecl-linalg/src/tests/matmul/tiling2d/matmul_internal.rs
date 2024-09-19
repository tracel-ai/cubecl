#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_tiling2d_internal {
    () => {
        #[test]
        pub fn tiling2d_matmul_outer_product_vectorized_test() {
            tests::tiling2d::compute_loop::tile_outer_product_vectorized_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn tiling2d_matmul_outer_product_vectorized_test_2() {
            tests::tiling2d::compute_loop::tile_outer_product_vectorized_unit_test_2::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn tiling2d_matmul_compute_loop_vectorized_test() {
            tests::tiling2d::compute_loop::compute_loop_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_lhs_transposed_unit_test() {
            tests::tiling2d::load_shared_memory::load_lhs_transposed_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_cube_test() {
            tests::tiling2d::load_shared_memory::load_lhs_transposed_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_plain_unit_test() {
            tests::tiling2d::load_shared_memory::load_lhs_plain_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_lhs_plain_out_of_bounds_unit_test() {
            tests::tiling2d::load_shared_memory::load_lhs_plain_out_of_bounds_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_out_of_bounds_cube_test() {
            tests::tiling2d::load_shared_memory::load_lhs_transposed_out_of_bounds_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_offset_cube_test() {
            tests::tiling2d::load_shared_memory::load_lhs_transposed_offset_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_plain_unit_test() {
            tests::tiling2d::load_shared_memory::load_rhs_plain_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_rhs_plain_cube_test() {
            tests::tiling2d::load_shared_memory::load_rhs_plain_cube_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_rhs_plain_cube_offset_test() {
            tests::tiling2d::load_shared_memory::load_rhs_plain_cube_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_transposed_unit_test() {
            tests::tiling2d::load_shared_memory::load_rhs_transposed_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_transposed_out_of_bounds_unit_test() {
            tests::tiling2d::load_shared_memory::load_rhs_transposed_out_of_bounds_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_over_height_unit_test() {
            tests::tiling2d::write_output::write_to_output_over_height_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_over_width_unit_test() {
            tests::tiling2d::write_output::write_to_output_over_width_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_vectorized_less_than_tile_unit_test() {
            tests::tiling2d::write_output::write_to_output_vectorized_less_than_tile_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_scalar_unit_test() {
            tests::tiling2d::write_output::write_to_output_scalar_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn write_to_output_scalar_out_of_bounds_cube_test() {
            tests::tiling2d::write_output::write_to_output_scalar_out_of_bounds_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }
    };
}

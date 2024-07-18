#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_matmul_internal {
    () => {
        use cubecl_lac::matmul::{
            cmma::{
                cmma_compute_loop_tests, cmma_load_shared_memory_tests, cmma_write_output_tests,
            },
            tiling2d::{
                compute_loop_tests, load_shared_memory_tests, outer_product_tests,
                write_output_tests,
            },
        };

        use super::*;

        #[test]
        pub fn tiling2d_matmul_outer_product_vectorized_test() {
            outer_product_tests::tile_outer_product_vectorized_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn tiling2d_matmul_outer_product_vectorized_test_2() {
            outer_product_tests::tile_outer_product_vectorized_unit_test_2::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn tiling2d_matmul_compute_loop_vectorized_test() {
            compute_loop_tests::compute_loop_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn tiling2d_matmul_compute_loop_unit_offset_test() {
            compute_loop_tests::compute_loop_unit_offset_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_lhs_transposed_unit_test() {
            load_shared_memory_tests::load_lhs_transposed_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_cube_test() {
            load_shared_memory_tests::load_lhs_transposed_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_plain_unit_test() {
            load_shared_memory_tests::load_lhs_plain_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_lhs_plain_out_of_bounds_unit_test() {
            load_shared_memory_tests::load_lhs_plain_out_of_bounds_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_out_of_bounds_cube_test() {
            load_shared_memory_tests::load_lhs_transposed_out_of_bounds_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_lhs_transposed_offset_cube_test() {
            load_shared_memory_tests::load_lhs_transposed_offset_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_plain_unit_test() {
            load_shared_memory_tests::load_rhs_plain_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_rhs_plain_cube_test() {
            load_shared_memory_tests::load_rhs_plain_cube_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn load_rhs_plain_cube_offset_test() {
            load_shared_memory_tests::load_rhs_plain_cube_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_transposed_unit_test() {
            load_shared_memory_tests::load_rhs_transposed_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn load_rhs_transposed_out_of_bounds_unit_test() {
            load_shared_memory_tests::load_rhs_transposed_out_of_bounds_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_over_height_unit_test() {
            write_output_tests::write_to_output_over_height_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_over_width_unit_test() {
            write_output_tests::write_to_output_over_width_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_vectorized_less_than_tile_unit_test() {
            write_output_tests::write_to_output_vectorized_less_than_tile_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn write_to_output_scalar_unit_test() {
            write_output_tests::write_to_output_scalar_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn write_to_output_scalar_out_of_bounds_cube_test() {
            write_output_tests::write_to_output_scalar_out_of_bounds_cube_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_compute_loop_k_test() {
            cmma_compute_loop_tests::compute_loop_k_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_compute_loop_warp_test() {
            cmma_compute_loop_tests::compute_loop_warp_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_compute_loop_cmma_offseted_warp_test() {
            cmma_compute_loop_tests::compute_loop_cmma_offseted_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_compute_loop_cmma_offseted_warp_in_shared_memory_test() {
            cmma_compute_loop_tests::compute_loop_cmma_offseted_warp_in_shared_memory_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_warp_test() {
            cmma_compute_loop_tests::cmma_warp_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_unit_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_unit_test() {
            cmma_load_shared_memory_tests::load_shared_memory_rhs_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_vertical_out_of_bound_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_vertical_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_horizontal_out_of_bound_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_horizontal_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_whole_out_of_bound_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_whole_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_rhs_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_second_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_second_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_rhs_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_third_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_third_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_third_warp_test() {
            cmma_load_shared_memory_tests::load_shared_memory_rhs_third_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_k_offset_test() {
            cmma_load_shared_memory_tests::load_shared_memory_lhs_k_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_k_offset_test() {
            cmma_load_shared_memory_tests::load_shared_memory_rhs_k_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_write_output_unit_test() {
            cmma_write_output_tests::cmma_write_output_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_warp_test() {
            cmma_write_output_tests::cmma_write_output_warp_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_second_warp_test() {
            cmma_write_output_tests::cmma_write_output_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_write_output_third_fourth_warps_test() {
            cmma_write_output_tests::cmma_write_output_third_fourth_warps_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_compute_loop_two_warps_same_tile_row_test() {
            cmma_compute_loop_tests::cmma_compute_loop_two_warps_same_tile_row_test::<TestRuntime>(
                &Default::default(),
            )
        }
    };
}

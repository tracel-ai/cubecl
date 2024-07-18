#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma_internal {
    () => {
        #[test]
        #[ignore] // does not work with n_tiles = 2 hardcoded
        pub fn cmma_compute_loop_k_test() {
            tests::cmma::compute_loop::compute_loop_k_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_compute_loop_warp_test() {
            tests::cmma::compute_loop::compute_loop_warp_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_compute_loop_two_warps_same_tile_row_test() {
            tests::cmma::compute_loop::cmma_compute_loop_two_warps_same_tile_row_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_unit_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_unit_test() {
            tests::cmma::load_shared_memory::load_shared_memory_rhs_unit_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_vertical_out_of_bound_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_vertical_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_horizontal_out_of_bound_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_horizontal_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_whole_out_of_bound_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_whole_out_of_bound_warp_test::<
                TestRuntime,
            >(&Default::default())
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_rhs_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_second_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_second_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_rhs_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_third_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_third_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_third_warp_test() {
            tests::cmma::load_shared_memory::load_shared_memory_rhs_third_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_lhs_k_offset_test() {
            tests::cmma::load_shared_memory::load_shared_memory_lhs_k_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_load_shared_memory_rhs_k_offset_test() {
            tests::cmma::load_shared_memory::load_shared_memory_rhs_k_offset_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_write_output_unit_test() {
            tests::cmma::write_output::cmma_write_output_unit_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_warp_test() {
            tests::cmma::write_output::cmma_write_output_warp_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_warp_horizontal_out_of_bounds_test() {
            tests::cmma::write_output::cmma_write_output_warp_horizontal_out_of_bounds_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_warp_vertical_out_of_bounds_test() {
            tests::cmma::write_output::cmma_write_output_warp_vertical_out_of_bounds_test::<TestRuntime>(&Default::default())
        }

        #[test]
        pub fn cmma_write_output_warp_whole_out_of_bounds_test() {
            tests::cmma::write_output::cmma_write_output_warp_whole_out_of_bounds_test::<TestRuntime>(&Default::default())
        }


        #[test]
        pub fn cmma_write_output_second_warp_test() {
            tests::cmma::write_output::cmma_write_output_second_warp_test::<TestRuntime>(
                &Default::default(),
            )
        }

        #[test]
        pub fn cmma_write_output_third_fourth_warps_test() {
            tests::cmma::write_output::cmma_write_output_third_fourth_warps_test::<TestRuntime>(
                &Default::default(),
            )
        }

    };
}

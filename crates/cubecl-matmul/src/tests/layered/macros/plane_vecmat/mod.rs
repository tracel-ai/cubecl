mod algorithm;
mod precision;
mod tiling_scheme;

#[macro_export]
macro_rules! testgen_matmul_plane_vecmat {
    () => {
        mod matmul_plane_vecmat {
            use super::*;
            use cubecl_matmul::components::tile::io::Filled;
            type TMM =
                $crate::components::tile::plane_vec_mat_inner_product::PlaneVecMatInnerProduct<
                    Filled,
                >;

            #[cfg(all(feature = "matmul_tests_plane", feature = "matmul_tests_vecmat"))]
            $crate::testgen_matmul_plane_vecmat_algorithm!();
        }
    };
}

mod partition;
mod stage;
mod tile;

#[macro_export]
macro_rules! testgen_matmul_plane_vecmat_tiling_scheme {
    ($algorithm: ty, $precision: ty) => {
        use $crate::components::TilingScheme;

        use super::*;

        $crate::testgen_matmul_plane_vecmat_tile!($algorithm, $precision, TilingScheme::builder());
    };
}

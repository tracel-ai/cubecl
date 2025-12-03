mod partition;
mod stage;
mod tile;

#[macro_export]
macro_rules! testgen_convolution_accelerated_tiling_scheme {
    ($algorithm: ty, $precision: ty) => {
        use cubecl_matmul::components::TilingScheme;

        use super::*;

        $crate::testgen_convolution_accelerated_tile!(
            $algorithm,
            $precision,
            TilingScheme::builder()
        );
    };
}

mod hypercube;
mod partition_buffering;
mod specialized;

#[macro_export]
macro_rules! testgen_matmul_advanced {
    ($kind: ident, $algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::components::{MatmulSelection, MatmulSelectionBuilder};

        mod _advanced {
            use super::*;

            pub fn get_selection_builder() -> MatmulSelectionBuilder {
                let tiling_scheme = $tiling_scheme_builder.build().unwrap();
                let client = TestRuntime::client(&Default::default());
                let plane_dim = client.properties().hardware.plane_size_max;
                MatmulSelection::builder(tiling_scheme, plane_dim)
            }
        }

        $crate::testgen_matmul_specialized!(
            $kind,
            $algorithm,
            $precision,
            _advanced::get_selection_builder()
        );
    };
}

mod hypercube;
mod partition_buffering;
mod specialized;

#[macro_export]
macro_rules! testgen_matmul_advanced {
    ($kind: ident, $algorithm: ty, $precision: ty, $tiling_scheme_builder: expr) => {
        use $crate::kernels::matmul::{MatmulSelection, MatmulSelectionBuilder};

        mod _advanced {
            use super::*;

            pub fn get_selection_builder() -> MatmulSelectionBuilder {
                let tiling_scheme = $tiling_scheme_builder.build().unwrap();
                let client = TestRuntime::client(&Default::default());
                let plane_dim = match client.properties().hardware.defined_plane_size() {
                    Some(val) => val,
                    None => {
                        panic!("Can't run test without a fixed plane size.");
                    }
                };
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

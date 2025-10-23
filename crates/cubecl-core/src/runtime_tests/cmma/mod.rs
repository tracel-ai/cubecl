mod cast;
mod manual;
mod scaled;
mod simple;
mod strided;

pub use cast::*;
pub use manual::*;
pub use scaled::*;
pub use simple::*;
pub use strided::*;

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cmma {
    () => {
        use super::*;
        use cubecl_core::prelude::*;

        #[test]
        fn test_cmma_simple_1() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1::<TestRuntime>(client, cube_dimensions);
        }

        #[test]
        fn test_cmma_simple_1_lined() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_lined::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_simple_1_lined_offset() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_1_lined_offset::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_simple_tf32() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_simple_tf32::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_f16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_f16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_cast_bf16() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_cast_bf16::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_strided() {
            let client = TestRuntime::client(&Default::default());
            let cube_dimensions = cube_dim::<TestRuntime>(&client);
            cubecl_core::runtime_tests::cmma::test_cmma_strided::<TestRuntime>(
                client,
                cube_dimensions,
            );
        }

        #[test]
        fn test_cmma_manual() {
            use cubecl_common::*;
            use half::{bf16, f16};

            fn test<
                A: CubeElement + Numeric,
                B: CubeElement + Numeric,
                CD: CubeElement + Numeric,
            >(
                m: usize,
                n: usize,
                k: usize,
            ) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_manual::<TestRuntime, A, B, CD>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                )
            }

            // CUDA
            test::<tf32, tf32, f32>(16, 8, 8);
            test::<f16, f16, f32>(16, 8, 16);
            test::<bf16, bf16, f32>(16, 8, 16);
            test::<e5m2, e5m2, f32>(16, 8, 32);
            test::<e4m3, e4m3, f32>(16, 8, 32);
            test::<e5m2, e4m3, f32>(16, 8, 32);
            test::<e4m3, e5m2, f32>(16, 8, 32);
            test::<i8, i8, i32>(16, 8, 32);
            test::<i8, u8, i32>(16, 8, 32);
            test::<u8, u8, i32>(16, 8, 32);
            test::<u8, i8, i32>(16, 8, 32);

            // HIP
            test::<f16, f16, f32>(16, 16, 16);
            test::<bf16, bf16, f32>(16, 16, 16);
        }

        #[test]
        fn test_cmma_scaled() {
            use cubecl_common::*;

            fn test<A: CubeElement + Numeric, B: CubeElement + Numeric>(
                m: usize,
                n: usize,
                k: usize,
                factor: usize,
            ) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_scaled::<TestRuntime, A, B>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                    factor,
                )
            }

            // FP4 needs more design for transferring properly as packed values
            test::<e5m2, e5m2>(16, 8, 32, 1);
            test::<e4m3, e4m3>(16, 8, 32, 1);
            test::<e5m2, e4m3>(16, 8, 32, 1);
            test::<e4m3, e5m2>(16, 8, 32, 1);
        }

        #[test]
        fn test_cmma_scaled_fp4() {
            use cubecl_common::*;

            fn test(m: usize, n: usize, k: usize, factor: usize) {
                let client = TestRuntime::client(&Default::default());
                let cube_dimensions = cube_dim::<TestRuntime>(&client);
                cubecl_core::runtime_tests::cmma::test_cmma_scaled_fp4::<TestRuntime>(
                    client,
                    cube_dimensions,
                    (m, n, k),
                    factor,
                )
            }

            // FP4 needs more design for transferring properly as packed values
            test(16, 8, 64, 2);
        }

        fn cube_dim<R: Runtime>(client: &ComputeClient<R::Server>) -> CubeDim {
            let plane_dim = client.properties().hardware.plane_size_max;
            CubeDim::new(plane_dim, 1, 1)
        }
    };
}

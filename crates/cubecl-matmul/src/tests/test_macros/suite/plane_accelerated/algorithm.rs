#[macro_export]
macro_rules! testgen_matmul_plane_accelerated_algorithm {
    () => {
        use $crate::kernels::matmul::{
            simple::SimpleAlgorithm,
            simple_barrier::SimpleBarrierAlgorithm,
            double_buffering::{CyclicDoubleBufferingAlgorithm, TilewiseDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm},
            ordered_double_buffering::OrderedDoubleBufferingAlgorithm
        };
        use $crate::components::global::load::{
            async_full_cyclic,
            async_full_maximize_slice_length,
            async_full_maximize_unit_count,
            sync_full_strided,
            sync_full_tilewise,
            async_full_cooperative,
        };
        use $crate::components::stage::{
            ColMajorTilingOrder,
            RowMajorTilingOrder
        };

        mod simple_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM>);
        }

        mod simple_strided {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM, sync_full_strided::LoadingStrategy, sync_full_strided::LoadingStrategy>);
        }

        mod simple_tilewise {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleAlgorithm<TMM, sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>, sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>>);
        }

        mod simple_barrier_cooperative {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_cooperative::LoadingStrategy>);
        }

        mod simple_barrier_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>>);
        }

        mod simple_barrier_maximize_slice_length {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_maximize_slice_length::LoadingStrategy>);
        }

        mod simple_barrier_maximize_unit_count {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(SimpleBarrierAlgorithm<TMM, async_full_maximize_unit_count::LoadingStrategy>);
        }

        mod double_buffering_cyclic {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(CyclicDoubleBufferingAlgorithm<TMM>);
        }

        mod double_buffering_tilewise {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(TilewiseDoubleBufferingAlgorithm<TMM>);
        }

        mod double_buffering_hybrid {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(HybridDoubleBufferingAlgorithm<TMM>);
        }

        mod ordered_double_buffering {
            use super::*;

            $crate::testgen_matmul_accelerated_precision!(OrderedDoubleBufferingAlgorithm<TMM>);
        }
    };
}

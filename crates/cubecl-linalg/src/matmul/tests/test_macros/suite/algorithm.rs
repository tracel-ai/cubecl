#[macro_export]
macro_rules! testgen_matmul_algorithm {
    // Select variant of cmma accelerated algorithm
    ($kind: ident) => {
        // use $crate::matmul::components::global::load::{
        //     async_full_cyclic, async_full_maximize_slice_length, async_full_maximize_unit_count, sync_full_strided, sync_full_tilewise, async_full_cooperative,
        // };
        // use $crate::matmul::components::stage::{ColMajorTilingOrder, RowMajorTilingOrder};
        // use $crate::matmul::kernels::matmul::double_buffering::{CyclicDoubleBufferingAlgorithm, TilewiseDoubleBufferingAlgorithm,
        //     HybridDoubleBufferingAlgorithm};
        // use $crate::matmul::kernels::matmul::ordered_double_buffering::OrderedDoubleBufferingAlgorithm;
        // use $crate::matmul::kernels::matmul::simple_barrier::SimpleBarrierAlgorithm;

        mod simple_cyclic {
            use super::*;
            use $crate::matmul::kernels::matmul::simple::SimpleAlgorithm;

            $crate::testgen_matmul_precision!($kind, SimpleAlgorithm<TMM>);
        }

        // mod simple_cyclic_multi_row {
        //     // TODO
        // }

        // mod simple_strided {
        //     $crate::testgen_matmul_precision!($kind, SimpleAlgorithm<TMM, sync_full_strided::LoadingStrategy, sync_full_strided::LoadingStrategy>);
        // }

        // mod simple_tilewise {
        //     $crate::testgen_matmul_precision!($kind, SimpleAlgorithm<TMM, sync_full_tilewise::LoadingStrategy<RowMajorTilingOrder>, sync_full_tilewise::LoadingStrategy<ColMajorTilingOrder>>);
        // }

        // #[test]
        // pub fn simple_barrier_cooperative() {
        //     $crate::testgen_matmul_precision!($kind, SimpleBarrierAlgorithm<TMM, async_full_cooperative::LoadingStrategy>);
        // }

        // #[test]
        // pub fn simple_barrier_cyclic() {
        //     $crate::testgen_matmul_precision!($kind, SimpleBarrierAlgorithm<TMM, async_full_cyclic::LoadingStrategy<ColMajorTilingOrder>>);
        // }

        // #[test]
        // pub fn simple_barrier_maximize_slice_length() {
        //     $crate::testgen_matmul_precision!($kind, SimpleBarrierAlgorithm<TMM, async_full_maximize_slice_length::LoadingStrategy>);
        // }

        // #[test]
        // pub fn simple_barrier_maximize_unit_count() {
        //     $crate::testgen_matmul_precision!($kind, SimpleBarrierAlgorithm<TMM, async_full_maximize_unit_count::LoadingStrategy>);
        // }

        // #[test]
        // pub fn double_buffering_single_row_cyclic() {
        //     $crate::testgen_matmul_precision!($kind, CyclicDoubleBufferingAlgorithm<TMM>);
        // }

        // #[test]
        // pub fn double_buffering_multi_row_cyclic() {
        //     // TODO
        // }

        // #[test]
        // pub fn double_buffering_single_row_tilewise() {
        //     $crate::testgen_matmul_precision!($kind, TilewiseDoubleBufferingAlgorithm<TMM>);
        // }

        // #[test]
        // pub fn double_buffering_multi_row_tilewise() {
        //     // TODO
        // }

        // #[test]
        // pub fn double_buffering_single_row_hybrid() {
        //     $crate::testgen_matmul_precision!($kind, HybridDoubleBufferingAlgorithm<TMM>);
        // }

        // #[test]
        // pub fn double_buffering_multi_row_hybrid() {
        //     // TODO
        // }

        // #[test]
        // pub fn ordered_double_buffering_single_row() {
        //     $crate::testgen_matmul_precision!($kind, OrderedDoubleBufferingAlgorithm<TMM>);
        // }
    };
}

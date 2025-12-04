#[macro_export]
macro_rules! testgen_convolution_partition_buffering {
    ($algorithm: ty, $precision: ty, $selection_builder: expr) => {
        use cubecl_matmul::components::stage::PartitionBuffering;

        #[cfg(not(feature = "conv_tests_partition_buffering"))]
        $crate::testgen_convolution_problem!(
            $algorithm,
            $precision,
            $selection_builder.partition_buffering(PartitionBuffering::Single)
        );

        #[cfg(feature = "conv_tests_partition_buffering")]
        mod pb1 {
            use super::*;

            $crate::testgen_convolution_problem!(
                $algorithm,
                $precision,
                $selection_builder.partition_buffering(PartitionBuffering::Single)
            );
        }

        #[cfg(feature = "conv_tests_partition_buffering")]
        mod pb2 {
            use super::*;

            $crate::testgen_convolution_problem!(
                $algorithm,
                $precision,
                $selection_builder.partition_buffering(PartitionBuffering::Double)
            );
        }
    };
}

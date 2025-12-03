#[macro_export]
macro_rules! testgen_convolution_problem_size {
    ($algorithm: ty, $precision: ty, $selection: expr) => {
        use $crate::tests::ConvolutionSize;

        mod g4x4x1x1 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 4,
                    w: 4,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod g17x17x1x1 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 17,
                    w: 17,
                    c: 1,
                    out_c: 1
                }
            );
        }

        mod g16x16x16x32 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 16,
                    w: 16,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod g32x32x32x16 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 32,
                    out_c: 16
                }
            );
        }

        mod g64x32x32x128 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 64,
                    w: 32,
                    c: 32,
                    out_c: 128
                }
            );
        }

        mod g32x32x64x3 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 32,
                    w: 32,
                    c: 64,
                    out_c: 3
                }
            );
        }

        // Deactivated, too long to run on cpu
        // mod g100x100x100x100 {
        //     use super::*;
        //     $crate::testgen_convolution_launch!(
        //         $tile,
        //         $partition,
        //         $stage,
        //         ConvolutionSize {
        //             h: 100,
        //             w: 100,
        //             c: 100,
        //             out_c: 100
        //         }
        //     );
        // }

        mod g20x20x16x32 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 20,
                    w: 20,
                    c: 16,
                    out_c: 32
                }
            );
        }

        mod g23x10x17x20 {
            use super::*;
            $crate::testgen_convolution_launch!(
                $algorithm,
                $precision,
                $selection,
                ConvolutionSize {
                    h: 23,
                    w: 10,
                    c: 17,
                    out_c: 20
                }
            );
        }
    };
}

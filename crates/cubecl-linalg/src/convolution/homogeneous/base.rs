use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use crate::{
    convolution::base::{Convolution, ConvolutionFamily, RuntimeArgs},
    matmul::components::{
        Ident,
        global::{
            GlobalConfig,
            args::{MatmulArgs, TensorInput, TensorInputIdent, TensorOutput},
        },
    },
};

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

#[cube(launch_unchecked)]
pub(crate) fn implicit_conv<
    Args: MatmulArgs,
    EI: Numeric,
    ES: Numeric,
    EA: Numeric,
    EO: Numeric,
    GMM: ConvolutionFamily,
>(
    inputs: &Input<Args, EI>,
    bias: &CubeOption<Tensor<Line<EO>>>,
    output: &mut Output<Args, EO>,
    runtime_args: RuntimeArgs,
    #[comptime] config: GMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorInput::<EI, EO, Args>::new(&state, TensorInputIdent::Lhs);
    let rhs = TensorInput::<EI, EO, Args>::new(&state, TensorInputIdent::Rhs);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let lhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    let x_offset = CUBE_POS_X * config.tiling_dimensions(Ident::Lhs).total_row();
    let y_offset = CUBE_POS_Y * config.tiling_dimensions(Ident::Rhs).total_col();
    let k_range = (0, runtime_args.size_k);

    let bias = match bias {
        CubeOption::Some(bias) => {
            CubeOption::new_Some(VirtualTensor::<EO>::new::<Tensor<Line<EO>>>(bias))
        }
        CubeOption::None => CubeOption::new_None(),
    };

    GMM::Convolution::<(EI, ES, EA, EO)>::execute(
        GMM::Convolution::<(EI, ES, EA, EO)>::init_lhs_loader(
            lhs,
            x_offset,
            k_range.0,
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_rhs_loader(
            rhs,
            k_range.0,
            y_offset,
            &runtime_args,
            config,
        ),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_bias_loader(bias, y_offset, config),
        GMM::Convolution::<(EI, ES, EA, EO)>::init_unloader(out, x_offset, y_offset),
        &mut GMM::Convolution::<(EI, ES, EA, EO)>::init_accumulator(config),
        k_range,
        config,
    );
}

pub mod config {
    use std::ops::Deref;

    use crate::{
        convolution::ConvGemmConfig,
        matmul::components::{MatmulConfig, MatrixLayout, TilingDimensions, global::GlobalConfig},
    };

    use super::*;

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct HomogeneousConfig<M: GlobalConfig> {
        matmul: M,
        kernel_size: (u32, u32),
        stride: (u32, u32),
        dilation: (u32, u32),
        padding: (i32, i32),
    }

    impl<M: GlobalConfig> Deref for HomogeneousConfig<M> {
        type Target = M;

        fn deref(&self) -> &Self::Target {
            &self.matmul
        }
    }

    impl<M: GlobalConfig> GlobalConfig for HomogeneousConfig<M> {
        type SmmConfig = M::SmmConfig;

        fn to_smm_config(&self) -> Self::SmmConfig {
            self.matmul.to_smm_config()
        }

        fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.matmul.global_line_size(ident)
        }

        fn stage_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.matmul.stage_line_size(ident)
        }

        fn tiling_dimensions<I: Into<Ident>>(&self, ident: I) -> TilingDimensions {
            self.matmul.tiling_dimensions(ident)
        }

        fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
            self.matmul.matrix_layout(ident)
        }

        fn num_planes(&self) -> u32 {
            self.matmul.num_planes()
        }

        fn plane_dim(&self) -> u32 {
            self.matmul.plane_dim()
        }

        fn check_row_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
            self.matmul.check_row_bounds(ident)
        }

        fn check_col_bounds<I: Into<Ident>>(&self, ident: I) -> bool {
            self.matmul.check_col_bounds(ident)
        }

        fn check_k_bounds(&self) -> bool {
            self.matmul.check_k_bounds()
        }
    }

    impl<M: GlobalConfig> ConvGemmConfig for HomogeneousConfig<M> {
        fn kernel_size(&self, dim: u32) -> u32 {
            match dim {
                0 => self.kernel_size.0,
                1 => self.kernel_size.1,
                _ => unreachable!(),
            }
        }

        fn dilation(&self, dim: u32) -> u32 {
            match dim {
                0 => self.dilation.0,
                1 => self.dilation.1,
                _ => unreachable!(),
            }
        }

        fn stride(&self, dim: u32) -> u32 {
            match dim {
                0 => self.stride.0,
                1 => self.stride.1,
                _ => unreachable!(),
            }
        }

        fn padding(&self, dim: u32) -> i32 {
            match dim {
                0 => self.padding.0,
                1 => self.padding.1,
                _ => unreachable!(),
            }
        }
    }

    impl<M: GlobalConfig> MatmulConfig for HomogeneousConfig<M> {}

    impl<M: GlobalConfig> HomogeneousConfig<M> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            matmul: M,
            kernel_size: (u32, u32),
            stride: (u32, u32),
            dilation: (u32, u32),
            padding: (i32, i32),
        ) -> Self {
            Self {
                matmul,
                kernel_size,
                stride,
                dilation,
                padding,
            }
        }

        pub fn to_matmul_config(self) -> M {
            self.matmul
        }
    }
}

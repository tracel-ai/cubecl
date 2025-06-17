use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand, FastDivmod, FastDivmodArgs,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

use crate::base::{Convolution, ConvolutionFamily, RuntimeArgs};

use cubecl_matmul::components::{
    Ident,
    global::{
        GlobalConfig,
        args::{MatmulArgs, TensorInput, TensorInputIdent, TensorOutput},
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

    let x_offset = CUBE_POS_X * config.tiling_scheme().elements_in_stage_m();
    let y_offset = CUBE_POS_Y * config.tiling_scheme().elements_in_stage_n();
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
        GMM::Convolution::<(EI, ES, EA, EO)>::init_writer(out, x_offset, y_offset),
        &mut GMM::Convolution::<(EI, ES, EA, EO)>::init_accumulator(config),
        k_range,
        config,
    );
}

pub(crate) fn shape_divmod<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &[usize],
) -> SequenceArg<'a, R, FastDivmod> {
    let shape = shape
        .iter()
        .map(|s| FastDivmodArgs::new(client, *s as u32))
        .collect();
    SequenceArg { values: shape }
}

pub mod config {
    use std::ops::Deref;

    use crate::{ConvGemmConfig, base::Dimensionality};
    use cubecl_matmul::{
        components::{
            InputIdent, MatmulConfig, MatmulLineSizes, MatrixLayout, TilingScheme,
            global::{
                GlobalConfig, PlaneRoleConfig, SpecializedLoadingSides, load::LoaderMode,
                multi_stage::EventLoadingMode,
            },
        },
        kernels::MatmulSetupError,
    };

    use super::*;

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct ConvolutionConfig<M: GlobalConfig> {
        matmul: M,
        kernel_size: [u32; 3],
        stride: [u32; 3],
        dilation: [u32; 3],
        padding: [i32; 3],
        dimensionality: Dimensionality,
        num_stages: u32,
    }

    impl<M: GlobalConfig> Deref for ConvolutionConfig<M> {
        type Target = M;

        fn deref(&self) -> &Self::Target {
            &self.matmul
        }
    }

    impl<M: GlobalConfig> GlobalConfig for ConvolutionConfig<M> {
        type StageConfig = M::StageConfig;

        fn stage_config(&self) -> Self::StageConfig {
            self.matmul.stage_config()
        }

        fn global_line_size<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.matmul.global_line_size(ident)
        }

        fn matrix_layout<I: Into<Ident>>(&self, ident: I) -> MatrixLayout {
            self.matmul.matrix_layout(ident)
        }

        fn num_loading_planes<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.matmul.num_loading_planes(ident)
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

        fn precompute_job(&self) -> bool {
            self.matmul.precompute_job()
        }

        fn num_stages(&self, _ident: InputIdent) -> u32 {
            self.num_stages
        }

        fn loader_mode(&self) -> LoaderMode {
            self.matmul.loader_mode()
        }

        fn tiling_scheme(&self) -> TilingScheme {
            self.matmul.tiling_scheme()
        }

        fn event_loading_mode(&self, ident: InputIdent) -> EventLoadingMode {
            self.matmul.event_loading_mode(ident)
        }

        fn plane_role_config(&self) -> PlaneRoleConfig {
            self.matmul.plane_role_config()
        }

        fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
            self.matmul.specialized_loading_sides()
        }

        fn cube_dim(&self) -> CubeDim {
            CubeDim::new(self.plane_dim(), self.tiling_scheme().tiles_in_stage_m(), 1)
        }
    }

    impl<M: GlobalConfig> ConvGemmConfig for ConvolutionConfig<M> {
        fn kernel_size(&self, dim: u32) -> u32 {
            self.kernel_size[dim as usize]
        }

        fn dilation(&self, dim: u32) -> u32 {
            self.dilation[dim as usize]
        }

        fn stride(&self, dim: u32) -> u32 {
            self.stride[dim as usize]
        }

        fn padding(&self, dim: u32) -> i32 {
            self.padding[dim as usize]
        }

        fn dimensionality(&self) -> Dimensionality {
            self.dimensionality
        }

        fn line_sizes(&self) -> cubecl_matmul::components::MatmulLineSizes {
            MatmulLineSizes {
                lhs: self.global_line_size(Ident::Lhs) as u8,
                rhs: self.global_line_size(Ident::Rhs) as u8,
                out: self.global_line_size(Ident::Out) as u8,
            }
        }
    }

    impl<M: GlobalConfig> MatmulConfig for ConvolutionConfig<M> {}

    impl<M: GlobalConfig> ConvolutionConfig<M> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            matmul: M,
            kernel_size: &[u32],
            stride: &[u32],
            dilation: &[u32],
            padding: &[i32],
            dim: Dimensionality,
            num_stages: u32,
        ) -> Result<Self, MatmulSetupError> {
            let dims = kernel_size.len();

            let mut this = Self {
                matmul,
                kernel_size: [0; 3],
                stride: [0; 3],
                dilation: [0; 3],
                padding: [0; 3],
                dimensionality: dim,
                num_stages,
            };
            this.kernel_size[0..dims].copy_from_slice(kernel_size);
            this.stride[0..dims].copy_from_slice(stride);
            this.dilation[0..dims].copy_from_slice(dilation);
            this.padding[0..dims].copy_from_slice(padding);
            Ok(this)
        }

        pub fn to_matmul_config(self) -> M {
            self.matmul
        }
    }
}

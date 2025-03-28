use crate::matmul::components::{
    Args, EG, Ident, MatmulLaunch, MatmulPrecision, MatmulSpec, TilingDimensions,
    config::MatmulConfig,
    global::{
        self, Quantization,
        args::{self, MatmulArgs, TensorInput, TensorOutput},
    },
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily: 'static + Send + Sync + MatmulLaunch<Config: BatchConfig> {
    type Matmul<MP: MatmulPrecision>: BatchMatmul<MP, Config = Self::Config>;
}

#[cube]
/// Provides matrix multiplication operations at the batch level.
///
/// At the batch level,
///  - Inputs are whole tensors in global memory.
///  - All Cubes can collaborate to solve the problem
///  - Dimensions M, N and K can be arbitrary large,
///    as well as the number of batches.
///
/// # Assumptions
/// - Line sizes of the inputs evenly divide the dimension they are aligned with.
/// - Enough Cubes are launched to perform the whole computation.
///
/// # Safety
///
/// It is not assumed that the matmul's dimensions match its inputs dimensions perfectly.
/// It is therefore important to use an underlying global matmul that performs check bounds,
/// and to not launch more Cubes than necessary.
pub trait BatchMatmul<MP: MatmulPrecision>: 'static + Send + Sync {
    type Config: BatchConfig;

    /// Performs batchwise matrix multiplication over tensors.
    fn execute(
        lhs: VirtualTensor<MP::EG>,
        rhs: VirtualTensor<MP::EG>,
        out: VirtualTensor<MP::EG, ReadWrite>,
        size_k: u32,
        quantization: CubeOption<Quantization<MP::EG>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig: MatmulConfig {
    /// Underlying Global matmul config
    type GmmConfig: global::GlobalConfig;

    /// Convert itself to the underlying global matmul config
    fn to_gmm_config(&self) -> Self::GmmConfig;

    /// Returns the [StageDim] for the given ident
    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions;

    /// Returns the largest m dimension supported with these configs
    fn max_m(&self) -> u32;

    /// Returns the largest n dimension supported with these configs
    fn max_n(&self) -> u32;

    /// Returns the largest number of batches supported with these configs
    fn max_batches(&self) -> u32;

    /// Returns true if the matmul is quantized.
    fn quantized(&self) -> bool;
}

type Input<MS> = <Args<MS> as MatmulArgs>::Input<EG<MS>>;
type Output<MS> = <Args<MS> as MatmulArgs>::Output<EG<MS>>;

#[cube(launch_unchecked)]
pub(crate) fn matmul<MS: MatmulSpec, BMM: BatchMatmulFamily>(
    inputs: &Input<MS>,
    output: &mut Output<MS>,
    size_k: u32,
    #[comptime] config: BMM::Config,
) {
    let mut state = MS::Args::init_state(inputs, output);

    let lhs = TensorInput::<EG<MS>, Args<MS>>::new(&state, args::TensorInputIdent::Lhs);
    let rhs = TensorInput::<EG<MS>, Args<MS>>::new(&state, args::TensorInputIdent::Rhs);
    let mut out = TensorOutput::<EG<MS>, Args<MS>>::new(&mut state);

    let lhs = VirtualTensor::<EG<MS>>::new::<TensorInput<EG<MS>, Args<MS>>>(&lhs);
    let rhs = VirtualTensor::<EG<MS>>::new::<TensorInput<EG<MS>, Args<MS>>>(&rhs);
    let out = VirtualTensor::<EG<MS>, ReadWrite>::new::<TensorOutput<EG<MS>, Args<MS>>>(&mut out);

    let quantization = if config.quantized() {
        CubeOption::new_Some(Args::<MS>::quantization(&state))
    } else {
        CubeOption::new_None()
    };

    BMM::Matmul::<MS::Precision>::execute(lhs, rhs, out, size_k, quantization, config);
}

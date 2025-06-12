use crate::{
    components::{
        InputRuntimeArg, MatmulLaunch, MatmulLineSizes, MatmulPrecision, MatmulProblem, MatmulSpec,
        OutputRuntimeArg, Quantized, TilingScheme,
        config::MatmulConfig,
        global::{
            self, GlobalConfig as _, Quantization,
            args::{self, MatmulArgs, TensorInput, TensorOutput},
        },
    },
    kernels::matmul::MatmulSelection,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

/// A family of [matmuls](BatchMatmul) working with any [precision](MatmulPrecision).
pub trait BatchMatmulFamily: 'static + Send + Sync {
    type Matmul<MP: MatmulPrecision>: BatchMatmul<MP, Config = Self::Config>;
    type Config: BatchConfig;
    type Input;

    fn cube_count(selection: &MatmulSelection, problem: &MatmulProblem) -> CubeCount;

    fn setup(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config;

    /// Entry point
    ///
    /// # Safety
    ///
    /// Out-of-bounds can happen
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    );
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
        lhs: VirtualTensor<MP::EI>,
        rhs: VirtualTensor<MP::EI>,
        out: VirtualTensor<MP::EO, ReadWrite>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    );
}

/// Configuration for the [batch matmul](BatchMatmul) level.
pub trait BatchConfig: MatmulConfig {
    /// Underlying Global matmul config
    type GlobalConfig: global::GlobalConfig;

    /// Convert itself to the underlying global matmul config
    fn global_config(&self) -> Self::GlobalConfig;

    /// Returns the largest m dimension supported with these configs
    fn max_problem_m(&self) -> u32;

    /// Returns the largest n dimension supported with these configs
    fn max_problem_n(&self) -> u32;

    /// Returns the largest number of batches supported with these configs
    fn max_problem_batches(&self) -> u32;

    /// Returns true if the matmul is quantized.
    fn quantized(&self) -> bool;

    fn tiling_scheme(&self) -> TilingScheme {
        self.global_config().tiling_scheme()
    }
}

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

#[cube(launch_unchecked)]
pub(crate) fn matmul<
    Args: MatmulArgs,
    EI: Numeric,
    ES: Numeric,
    EA: Numeric,
    EO: Numeric,
    BMM: BatchMatmulFamily,
>(
    inputs: &Input<Args, EI>,
    output: &mut Output<Args, EO>,
    #[comptime] config: BMM::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Lhs);
    let rhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Rhs);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let lhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    if config.quantized() {
        let quantization = Args::quantization::<(EI, ES, EA, EO, Quantized)>(&state);
        BMM::Matmul::<(EI, ES, EA, EO, Quantized)>::execute(
            lhs,
            rhs,
            out,
            CubeOption::new_Some(quantization),
            config,
        );
    } else {
        BMM::Matmul::<(EI, ES, EA, EO)>::execute(lhs, rhs, out, CubeOption::new_None(), config);
    };
}

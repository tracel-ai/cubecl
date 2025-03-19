use std::marker::PhantomData;

use crate::matmul::components::global::{GlobalMatmul, GlobalMatmulFamily};
use crate::matmul::components::{batch, Ident, MatmulConfigFactory, MatmulLaunch};
use crate::matmul::components::{
    InputRuntimeArg, InvalidConfigError, MatmulPrecision, MatmulProblem, MatmulSpec,
    OutputRuntimeArg,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use batch::{BatchMatmul, BatchMatmulFamily};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::{gmm_execute_tma, one_to_one::Config, BatchConfig as _, CubeDispatch};

pub struct OneToOneTmaMatmulFamily<GMM: GlobalMatmulFamily, C: CubeDispatch> {
    _gmm: PhantomData<GMM>,
    _c: PhantomData<C>,
}

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> BatchMatmulFamily
    for OneToOneTmaMatmulFamily<GMM, C>
{
    type Matmul<MP: MatmulPrecision> = OneToOneTmaMatmul<MP, GMM::Matmul<MP>, C>;
}

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> MatmulConfigFactory
    for OneToOneTmaMatmulFamily<GMM, C>
{
    type Input = GMM::Input;
    type Config = Config<GMM::Config, C>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        GMM::check_config(&config.to_gmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        GMM::check_availability::<R, MP>(client, &config.to_gmm_config())
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let gmm_config = GMM::make_config(input, problem, cube_dim, cube_count, quantized);
        let cube_count = if let CubeCount::Static(x, y, z) = cube_count {
            (*x, *y, *z)
        } else {
            panic!("Dynamic cube count unsupported")
        };

        Config::<GMM::Config, C>::new(gmm_config, cube_count)
    }
}

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> MatmulLaunch for OneToOneTmaMatmulFamily<GMM, C> {
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        size_k: ScalarArg<u32>,
        config: Self::Config,
    ) {
        super::matmul::launch_unchecked::<MS::EG, MS::ES, MS::EA, MS::Args, Self, R>(
            client, cube_count, cube_dim, input, output, size_k, config,
        );
    }
}

/// Executes matrix multiplication at the batch level,
/// assigning each cube to a single global matmul.
///
/// Note: This algorithm requires one cube per global matmul;
/// insufficient cubes will result in incomplete computations.
pub struct OneToOneTmaMatmul<MP: MatmulPrecision, GMM: GlobalMatmul<MP>, C: CubeDispatch> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _c: PhantomData<C>,
}

#[cube]
impl<MP: MatmulPrecision, GMM: GlobalMatmul<MP>, C: CubeDispatch> BatchMatmul<MP>
    for OneToOneTmaMatmul<MP, GMM, C>
{
    type Config = Config<GMM::Config, C>;

    fn execute(
        lhs: VirtualTensor<MP::EG>,
        rhs: VirtualTensor<MP::EG>,
        out: VirtualTensor<MP::EG, ReadWrite>,
        size_k: u32,
        #[comptime] config: Self::Config,
    ) {
        let (x_index, y_index) = C::x_y_indices();
        let x_offset = x_index * config.tiling_dimensions(Ident::Lhs).total_row();
        let y_offset = y_index * config.tiling_dimensions(Ident::Rhs).total_col();
        let nth_batch = C::batch_index();
        let k_range = (0u32, size_k);

        let gmm_config = config.to_gmm_config();

        gmm_execute_tma::<MP, GMM>(
            lhs,
            rhs,
            out,
            x_offset,
            y_offset,
            nth_batch,
            &mut GMM::init_accumulator(gmm_config),
            k_range,
            gmm_config,
        );
    }
}

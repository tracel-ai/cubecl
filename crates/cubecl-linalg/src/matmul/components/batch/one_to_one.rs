use std::marker::PhantomData;

use crate::matmul::components::batch::shared::gmm_execute;
use crate::matmul::components::global::{GlobalMatmul, GlobalMatmulFamily};
use crate::matmul::components::{
    Ident, MatmulConfigFactory, MatmulLaunch, TilingDimensions, batch, config::MatmulConfig, global,
};
use crate::matmul::components::{
    InputRuntimeArg, InvalidConfigError, MatmulPrecision, MatmulProblem, MatmulSpec,
    OutputRuntimeArg,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use batch::{BatchMatmul, BatchMatmulFamily};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use super::{BatchConfig as _, CubeDispatch};

pub struct OneToOneMatmulFamily<GMM: GlobalMatmulFamily, C: CubeDispatch> {
    _gmm: PhantomData<GMM>,
    _c: PhantomData<C>,
}

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> BatchMatmulFamily for OneToOneMatmulFamily<GMM, C> {
    type Matmul<MP: MatmulPrecision> = OneToOneMatmul<MP, GMM::Matmul<MP>, C>;
}

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> MatmulConfigFactory
    for OneToOneMatmulFamily<GMM, C>
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
        GMM::check_availability::<R, MP>(client, &config.gmm_config)
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

impl<GMM: GlobalMatmulFamily, C: CubeDispatch> MatmulLaunch for OneToOneMatmulFamily<GMM, C> {
    unsafe fn launch_unchecked<'a, MS: MatmulSpec, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: InputRuntimeArg<'a, MS, R>,
        output: OutputRuntimeArg<'a, MS, R>,
        config: Self::Config,
    ) {
        unsafe {
            super::matmul::launch_unchecked::<MS::EG, MS::ES, MS::EA, MS::Args, Self, R>(
                client, cube_count, cube_dim, input, output, config,
            );
        }
    }
}

/// Executes matrix multiplication at the batch level,
/// assigning each cube to a single global matmul.
///
/// Note: This algorithm requires one cube per global matmul;
/// insufficient cubes will result in incomplete computations.
pub struct OneToOneMatmul<MP: MatmulPrecision, GMM: GlobalMatmul<MP>, C: CubeDispatch> {
    _mp: PhantomData<MP>,
    _gmm: PhantomData<GMM>,
    _c: PhantomData<C>,
}

#[cube]
impl<MP: MatmulPrecision, GMM: GlobalMatmul<MP>, C: CubeDispatch> BatchMatmul<MP>
    for OneToOneMatmul<MP, GMM, C>
{
    type Config = Config<GMM::Config, C>;

    fn execute(
        lhs: VirtualTensor<MP::EG>,
        rhs: VirtualTensor<MP::EG>,
        out: VirtualTensor<MP::EG, ReadWrite>,
        #[comptime] config: Self::Config,
    ) {
        let (x_index, y_index) = C::x_y_indices();
        let x_offset = x_index * config.tiling_dimensions(Ident::Lhs).total_row();
        let y_offset = y_index * config.tiling_dimensions(Ident::Rhs).total_col();
        let nth_batch = C::batch_index();
        let rank = lhs.rank();
        let k_range = (0, lhs.shape(rank - 1));

        let gmm_config = config.to_gmm_config();

        gmm_execute::<MP, GMM>(
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

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct Config<G: global::GlobalConfig, C: CubeDispatch> {
    gmm_config: G,
    cube_count: (u32, u32, u32),
    _c: PhantomData<C>,
}

impl<G: global::GlobalConfig, C: CubeDispatch> batch::BatchConfig for Config<G, C> {
    type GmmConfig = G;

    fn to_gmm_config(&self) -> Self::GmmConfig {
        self.gmm_config
    }

    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions {
        self.gmm_config.tiling_dimensions(ident)
    }

    fn max_m(&self) -> u32 {
        C::max_x(self.cube_count) * self.tiling_dimensions(Ident::Out).total_row()
    }

    fn max_n(&self) -> u32 {
        C::max_y(self.cube_count) * self.tiling_dimensions(Ident::Out).total_col()
    }

    fn max_batches(&self) -> u32 {
        C::max_batches(self.cube_count)
    }
}

impl<G: global::GlobalConfig, C: CubeDispatch> MatmulConfig for Config<G, C> {}

impl<G: global::GlobalConfig, C: CubeDispatch> Config<G, C> {
    pub fn new(gmm_config: G, cube_count: (u32, u32, u32)) -> Self {
        Self {
            gmm_config,
            cube_count,
            _c: PhantomData,
        }
    }
}

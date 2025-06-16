use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};

use crate::{
    components::{
        Ident, MatmulConfig, MatmulLineSizes, MatmulPrecision, MatmulProblem,
        batch::{BatchConfig, Partitioner},
        global::GlobalConfig,
    },
    kernels::MatmulSetupError,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PartitionedBatchConfig<G: GlobalConfig, P: Partitioner> {
    global_config: G,
    _phantom: PhantomData<P>,
}

impl<G: GlobalConfig, P: Partitioner> BatchConfig for PartitionedBatchConfig<G, P> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn quantized(&self) -> bool {
        self.global_config().quantized()
    }

    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn line_sizes(&self) -> MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.global_config.global_line_size(Ident::Lhs) as u8,
            rhs: self.global_config.global_line_size(Ident::Rhs) as u8,
            out: self.global_config.global_line_size(Ident::Out) as u8,
        }
    }

    fn cube_count(&self, problem: &MatmulProblem) -> CubeCount {
        let tiling_scheme = self.tiling_scheme();
        let elements_in_m = tiling_scheme.elements_in_global_partition_m();
        let elements_in_n = tiling_scheme.elements_in_global_partition_n();

        let (x, y, z) = P::create_cube_count(
            (problem.m as u32).div_ceil(elements_in_m),
            (problem.n as u32).div_ceil(elements_in_n),
            (problem.num_batches() as u32).div_ceil(tiling_scheme.global_partition_size.batches),
        );

        CubeCount::Static(x, y, z)
    }
}

impl<G: GlobalConfig, P: Partitioner> MatmulConfig for PartitionedBatchConfig<G, P> {}

impl<G: GlobalConfig, P: Partitioner> PartitionedBatchConfig<G, P> {
    pub fn new<MP: MatmulPrecision, R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        global_config: G,
    ) -> Result<Self, MatmulSetupError> {
        Self {
            global_config,
            _phantom: PhantomData,
        }
        .validate()
    }

    fn validate(self) -> Result<Self, MatmulSetupError> {
        Ok(self)
    }
}

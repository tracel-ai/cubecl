use std::marker::PhantomData;

use cubecl_core::{CubeCount, CubeDim};

use crate::components::{
    MatmulConfig,
    batch::{BatchConfig, matmul::partitioner::Partitioner},
    global::GlobalConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the OneToOneBatchMatmul
pub struct PartitionedBatchConfig<G: GlobalConfig, P: Partitioner> {
    global_config: G,
    cube_count: (u32, u32, u32),
    _phantom: PhantomData<P>,
}

impl<G: GlobalConfig, P: Partitioner> BatchConfig for PartitionedBatchConfig<G, P> {
    type GlobalConfig = G;

    fn global_config(&self) -> Self::GlobalConfig {
        self.global_config
    }

    fn max_problem_m(&self) -> u32 {
        self.cube_count_m()
            * self
                .global_config
                .tiling_scheme()
                .elements_in_global_partition_m()
    }

    fn max_problem_n(&self) -> u32 {
        self.cube_count_n()
            * self
                .global_config
                .tiling_scheme()
                .elements_in_global_partition_n()
    }

    fn max_problem_batches(&self) -> u32 {
        self.cube_count_batch()
            * self
                .global_config
                .tiling_scheme()
                .global_partition_size
                .batches
    }

    fn quantized(&self) -> bool {
        self.global_config().quantized()
    }

    fn cube_dim(&self) -> CubeDim {
        self.global_config.cube_dim()
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static(self.cube_count.0, self.cube_count.1, self.cube_count.2)
    }
}

impl<G: GlobalConfig, P: Partitioner> MatmulConfig for PartitionedBatchConfig<G, P> {}

impl<G: GlobalConfig, P: Partitioner> PartitionedBatchConfig<G, P> {
    pub fn new(global_config: G, cube_count: (u32, u32, u32)) -> Self {
        Self {
            global_config,
            cube_count,
            _phantom: PhantomData,
        }
    }

    fn cube_count_m(&self) -> u32 {
        P::cube_count_m(self.cube_count)
    }

    fn cube_count_n(&self) -> u32 {
        P::cube_count_n(self.cube_count)
    }

    fn cube_count_batch(&self) -> u32 {
        P::cube_count_batches(self.cube_count)
    }
}

use crate::{
    components::{
        LoadSpecializationConfig, TilingScheme,
        global::load::LoaderMode,
        stage::{PartitionBuffering, StageVectorization},
    },
    kernels::matmul::LoadingPrecomputeStrategy,
};

#[derive(Debug)]
pub struct MatmulSelection {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
    pub stage_vectorization: StageVectorization,
    pub quantized: bool,
    pub partition_buffering: PartitionBuffering,
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    pub loader_mode: LoaderMode,
    pub load_specialization_config: LoadSpecializationConfig,
}

pub struct MatmulSelectionBuilder {
    plane_dim: Option<u32>,
    tiling_scheme: Option<TilingScheme>,
    stage_vectorization: StageVectorization,
    quantized: bool,
    partition_buffering: PartitionBuffering,
    loading_precompute_strategy: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
    load_specialization_config: LoadSpecializationConfig,
}

impl MatmulSelectionBuilder {
    pub fn new() -> Self {
        Self {
            plane_dim: None,
            tiling_scheme: None,
            stage_vectorization: StageVectorization::default(),
            quantized: false,
            partition_buffering: PartitionBuffering::default(),
            loading_precompute_strategy: LoadingPrecomputeStrategy::default(),
            loader_mode: LoaderMode::default(),
            load_specialization_config: LoadSpecializationConfig::default(),
        }
    }

    pub fn plane_dim(mut self, dim: u32) -> Self {
        self.plane_dim = Some(dim);
        self
    }

    pub fn tiling_scheme(mut self, scheme: TilingScheme) -> Self {
        self.tiling_scheme = Some(scheme);
        self
    }

    pub fn stage_vectorization(mut self, vec: StageVectorization) -> Self {
        self.stage_vectorization = vec;
        self
    }

    pub fn quantized(mut self, val: bool) -> Self {
        self.quantized = val;
        self
    }

    pub fn partition_buffering(mut self, buffering: PartitionBuffering) -> Self {
        self.partition_buffering = buffering;
        self
    }

    pub fn loading_precompute_strategy(mut self, strat: LoadingPrecomputeStrategy) -> Self {
        self.loading_precompute_strategy = strat;
        self
    }

    pub fn loader_mode(mut self, mode: LoaderMode) -> Self {
        self.loader_mode = mode;
        self
    }

    pub fn load_specialization_config(mut self, config: LoadSpecializationConfig) -> Self {
        self.load_specialization_config = config;
        self
    }

    pub fn build(self) -> Result<MatmulSelection, &'static str> {
        Ok(MatmulSelection {
            plane_dim: self.plane_dim.ok_or("plane_dim must be set")?,
            tiling_scheme: self.tiling_scheme.ok_or("tiling_scheme must be set")?,
            stage_vectorization: self.stage_vectorization,
            quantized: self.quantized,
            partition_buffering: self.partition_buffering,
            loading_precompute_strategy: self.loading_precompute_strategy,
            loader_mode: self.loader_mode,
            load_specialization_config: self.load_specialization_config,
        })
    }
}

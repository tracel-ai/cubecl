use crate::components::{
    TilingScheme,
    batch::HypercubeSelection,
    global::{LoadSpecializationConfig, load::LoaderMode},
    stage::{PartitionBuffering, StageVectorization},
};

#[derive(Debug, Clone)]
pub struct MatmulSelection {
    pub plane_dim: u32,
    pub tiling_scheme: TilingScheme,
    pub stage_vectorization: StageVectorization,
    pub quantized: bool,
    pub partition_buffering: PartitionBuffering,
    pub loading_precompute_strategy: LoadingPrecomputeStrategy,
    pub loader_mode: LoaderMode,
    pub load_specialization_config: LoadSpecializationConfig,
    pub hypercube_selection: HypercubeSelection,
}

impl MatmulSelection {
    pub fn builder(tiling_scheme: TilingScheme, plane_dim: u32) -> MatmulSelectionBuilder {
        let hypercube_config = HypercubeSelection::builder(&tiling_scheme).build();
        MatmulSelectionBuilder::new()
            .tiling_scheme(tiling_scheme)
            .hypercube_config(hypercube_config)
            .plane_dim(plane_dim)
    }
}

pub struct MatmulSelectionBuilder {
    plane_dim: Option<u32>,
    pub tiling_scheme: Option<TilingScheme>,
    hypercube_selection: Option<HypercubeSelection>,
    stage_vectorization: StageVectorization,
    quantized: bool,
    partition_buffering: PartitionBuffering,
    loading_precompute_strategy: LoadingPrecomputeStrategy,
    loader_mode: LoaderMode,
    load_specialization_config: LoadSpecializationConfig,
}

impl MatmulSelectionBuilder {
    fn new() -> Self {
        Self {
            plane_dim: None,
            tiling_scheme: None,
            hypercube_selection: None,
            stage_vectorization: StageVectorization::default(),
            quantized: false,
            partition_buffering: PartitionBuffering::default(),
            loading_precompute_strategy: LoadingPrecomputeStrategy::default(),
            loader_mode: LoaderMode::default(),
            load_specialization_config: LoadSpecializationConfig::default(),
        }
    }

    pub fn plane_dim(mut self, plane_dim: u32) -> Self {
        self.plane_dim = Some(plane_dim);
        self
    }

    pub fn tiling_scheme(mut self, tiling_scheme: TilingScheme) -> Self {
        self.tiling_scheme = Some(tiling_scheme);
        self
    }

    pub fn hypercube_config(mut self, hypercube_config: HypercubeSelection) -> Self {
        self.hypercube_selection = Some(hypercube_config);
        self
    }

    pub fn stage_vectorization(mut self, stage_vectorization: StageVectorization) -> Self {
        self.stage_vectorization = stage_vectorization;
        self
    }

    pub fn quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    pub fn partition_buffering(mut self, partition_buffering: PartitionBuffering) -> Self {
        self.partition_buffering = partition_buffering;
        self
    }

    pub fn loading_precompute_strategy(
        mut self,
        loading_precompute_strategy: LoadingPrecomputeStrategy,
    ) -> Self {
        self.loading_precompute_strategy = loading_precompute_strategy;
        self
    }

    pub fn loader_mode(mut self, loader_mode: LoaderMode) -> Self {
        self.loader_mode = loader_mode;
        self
    }

    pub fn load_specialization_config(
        mut self,
        load_specialization_config: LoadSpecializationConfig,
    ) -> Self {
        self.load_specialization_config = load_specialization_config;
        self
    }

    pub fn build(self) -> MatmulSelection {
        MatmulSelection {
            plane_dim: self.plane_dim.unwrap(),
            tiling_scheme: self.tiling_scheme.unwrap(),
            hypercube_selection: self.hypercube_selection.unwrap(),
            stage_vectorization: self.stage_vectorization,
            quantized: self.quantized,
            partition_buffering: self.partition_buffering,
            loading_precompute_strategy: self.loading_precompute_strategy,
            loader_mode: self.loader_mode,
            load_specialization_config: self.load_specialization_config,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum MultiRowStrategy {
    /// Always one row per plane
    #[default]
    Never,
    /// Always multiple rows per plane
    Always(u32),
    /// Uses multiple rows if the `m` dimension of the matmul implies at least the minimum number of stages along `m`
    Adaptive { minimum_stage_count: u32 },
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadingPrecomputeStrategy {
    /// Don't precompute anything in loading jobs
    #[default]
    Never,
    /// Precompute values that are shared across tasks
    Always,
}

impl From<LoadingPrecomputeStrategy> for bool {
    fn from(strategy: LoadingPrecomputeStrategy) -> Self {
        match strategy {
            LoadingPrecomputeStrategy::Always => true,
            LoadingPrecomputeStrategy::Never => false,
        }
    }
}

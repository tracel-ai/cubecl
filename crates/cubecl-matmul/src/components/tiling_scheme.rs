use super::size::{GlobalPartitionSize, MatmulDim, PartitionSize, StageSize, TileSize};

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
/// Complete tiling configuration for a matmul.
/// Encodes all structural information needed to compute tiling shapes and counts.
pub struct TilingScheme {
    pub tile_size: TileSize,
    pub partition_size: PartitionSize,
    pub stage_size: StageSize,
    pub global_partition_size: GlobalPartitionSize,
}

impl TilingScheme {
    /// Create a builder for TilingScheme
    pub fn builder() -> TilingSchemeBuilder {
        TilingSchemeBuilder::default()
    }
}

#[derive(Debug, Default)]
/// Builder for [`TilingScheme`]. Allows step-by-step configuration.
pub struct TilingSchemeBuilder {
    tile_size: Option<TileSize>,
    partition_size: Option<PartitionSize>,
    stage_size: Option<StageSize>,
    global_partition_size: Option<GlobalPartitionSize>,
}

impl TilingSchemeBuilder {
    /// Specify tile size for tiling scheme
    pub fn with_tile_size(mut self, tile_size: TileSize) -> Self {
        self.tile_size = Some(tile_size);
        self
    }

    /// Specify partition size for tiling scheme
    pub fn with_partition_size(mut self, partition_size: PartitionSize) -> Self {
        self.partition_size = Some(partition_size);
        self
    }

    /// Specify stage size for tiling scheme
    ///
    /// Only stage size k = 1 is supported
    pub fn with_stage_size(mut self, stage_size: StageSize) -> Self {
        assert!(stage_size.k == 1, "Stage size k > 1 is not supported");
        self.stage_size = Some(stage_size);
        self
    }

    /// Optional: specify global partition size for tiling scheme
    ///
    /// If not specified, will default to (1, 1, 1)
    pub fn with_global_partition_size(
        mut self,
        global_partition_size: GlobalPartitionSize,
    ) -> Self {
        self.global_partition_size = Some(global_partition_size);
        self
    }

    /// Finish building
    pub fn build(self) -> Result<TilingScheme, &'static str> {
        Ok(TilingScheme {
            tile_size: self.tile_size.ok_or("Missing tile_size")?,
            partition_size: self.partition_size.ok_or("Missing tiles_per_partition")?,
            stage_size: self.stage_size.ok_or("Missing partitions_per_stage")?,
            global_partition_size: self
                .global_partition_size
                .unwrap_or(GlobalPartitionSize::new(1, 1, 1)),
        })
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum TilingLevel {
    GlobalPartition,
    Stage,
    StagePartition,
    Tile,
    Element,
}

impl TilingScheme {
    fn try_count_1d(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        dim: MatmulDim,
    ) -> Option<u32> {
        use TilingLevel::*;

        match (child_level, parent_level) {
            (child, parent) if child == parent => Some(1),

            (Stage, GlobalPartition) => match dim {
                MatmulDim::M => Some(self.global_partition_size.m),
                MatmulDim::N => Some(self.global_partition_size.n),
                MatmulDim::K => None,
            },

            (StagePartition, Stage) => Some(self.stage_size.get(dim)),

            (Tile, StagePartition) => Some(self.partition_size.get(dim)),

            (Element, Tile) => Some(self.tile_size.get(dim)),

            (StagePartition, GlobalPartition) => Some(
                self.try_count_1d(StagePartition, Stage, dim)?
                    * self.try_count_1d(Stage, GlobalPartition, dim)?,
            ),

            (Tile, GlobalPartition) => Some(
                self.try_count_1d(Tile, Stage, dim)?
                    * self.try_count_1d(Stage, GlobalPartition, dim)?,
            ),

            (Element, GlobalPartition) => Some(
                self.try_count_1d(Element, Stage, dim)?
                    * self.try_count_1d(Stage, GlobalPartition, dim)?,
            ),

            (Tile, Stage) => Some(
                self.try_count_1d(StagePartition, Stage, dim)?
                    * self.try_count_1d(Tile, StagePartition, dim)?,
            ),

            (Element, Stage) => {
                Some(self.try_count_1d(Tile, Stage, dim)? * self.try_count_1d(Element, Tile, dim)?)
            }

            (Element, StagePartition) => Some(
                self.try_count_1d(Tile, StagePartition, dim)?
                    * self.try_count_1d(Element, Tile, dim)?,
            ),

            // Invalid transitions
            _ => None,
        }
    }

    fn count_1d(&self, child_level: TilingLevel, parent_level: TilingLevel, dim: MatmulDim) -> u32 {
        self.try_count_1d(child_level, parent_level, dim)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    pub fn partitions_per_stage_along_m(&self) -> u32 {
        self.count_1d(
            TilingLevel::StagePartition,
            TilingLevel::Stage,
            MatmulDim::M,
        )
    }

    pub fn partitions_per_stage_along_n(&self) -> u32 {
        self.count_1d(
            TilingLevel::StagePartition,
            TilingLevel::Stage,
            MatmulDim::N,
        )
    }

    pub fn elements_per_stage_along_m(&self) -> u32 {
        self.count_1d(TilingLevel::Element, TilingLevel::Stage, MatmulDim::M)
    }

    pub fn elements_per_stage_along_n(&self) -> u32 {
        self.count_1d(TilingLevel::Element, TilingLevel::Stage, MatmulDim::N)
    }

    pub fn elements_per_stage_along_k(&self) -> u32 {
        self.count_1d(TilingLevel::Element, TilingLevel::Stage, MatmulDim::K)
    }

    pub fn tiles_per_stage_partition_along_n(&self) -> u32 {
        self.count_1d(TilingLevel::Tile, TilingLevel::StagePartition, MatmulDim::N)
    }

    pub fn elements_per_global_partition_along_m(&self) -> u32 {
        self.count_1d(
            TilingLevel::Element,
            TilingLevel::GlobalPartition,
            MatmulDim::M,
        )
    }

    pub fn elements_per_global_partition_along_n(&self) -> u32 {
        self.count_1d(
            TilingLevel::Element,
            TilingLevel::GlobalPartition,
            MatmulDim::N,
        )
    }
}

#[derive(Debug, Clone)]
pub struct TilingScheme {
    tile_shape: TileShape,
    tiles_per_partition: TilesPerPartition,
    partitions_per_stage: PartitionsPerStage,
    stage_k_tile_count: u32,
}

impl TilingScheme {
    pub fn builder() -> TilingSchemeBuilder {
        TilingSchemeBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct TilingSchemeBuilder {
    tile_shape: Option<TileShape>,
    tiles_per_partition: Option<TilesPerPartition>,
    partitions_per_stage: Option<PartitionsPerStage>,
    stage_k_tile_count: Option<u32>,
}

impl TilingSchemeBuilder {
    pub fn with_tile_shape(mut self, tile_shape: TileShape) -> Self {
        self.tile_shape = Some(tile_shape);
        self
    }

    pub fn with_tiles_per_partition(mut self, tiles_per_partition: TilesPerPartition) -> Self {
        self.tiles_per_partition = Some(tiles_per_partition);
        self
    }

    pub fn with_partitions_per_stage(mut self, partitions_per_stage: PartitionsPerStage) -> Self {
        self.partitions_per_stage = Some(partitions_per_stage);
        self
    }

    pub fn with_stage_k_tile_count(mut self, stage_k_tile_count: u32) -> Self {
        self.stage_k_tile_count = Some(stage_k_tile_count);
        self
    }

    pub fn build(self) -> Result<TilingScheme, &'static str> {
        Ok(TilingScheme {
            tile_shape: self.tile_shape.ok_or("Missing tile_shape")?,
            tiles_per_partition: self
                .tiles_per_partition
                .ok_or("Missing tiles_per_partition")?,
            partitions_per_stage: self
                .partitions_per_stage
                .ok_or("Missing partitions_per_stage")?,
            stage_k_tile_count: self
                .stage_k_tile_count
                .ok_or("Missing stage_k_tile_count")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MatmulDim {
    M,
    N,
    K,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TilingLevel {
    Stage,
    Partition,
    Tile,
    Element,
}

impl TilingScheme {
    pub fn count_dim(
        &self,
        parent_level: TilingLevel,
        child_level: TilingLevel,
        dim: MatmulDim,
    ) -> u32 {
        self.try_count_dim(parent_level, child_level, dim)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    pub fn try_count_dim(
        &self,
        parent_level: TilingLevel,
        child_level: TilingLevel,
        dim: MatmulDim,
    ) -> Option<u32> {
        use MatmulDim::*;
        use TilingLevel::*;

        match (parent_level, child_level) {
            (parent, child) if parent == child => Some(1),

            (Stage, Partition) => Some(match dim {
                M => self.partitions_per_stage.m,
                N => self.partitions_per_stage.n,
                K => 1,
            }),

            (Partition, Tile) => Some(match dim {
                M => self.tiles_per_partition.m,
                N => self.tiles_per_partition.n,
                K => 1,
            }),

            (Tile, Element) => Some(match dim {
                M => self.tile_shape.m,
                N => self.tile_shape.n,
                K => self.tile_shape.k,
            }),

            (Stage, Tile) => {
                let partitions_per_stage = self.try_count_dim(Stage, Partition, dim)?;
                let tiles_per_partition = self.try_count_dim(Partition, Tile, dim)?;

                // We must account for the k dim which is not considered in partitions
                if let K = dim {
                    Some(self.stage_k_tile_count * partitions_per_stage * tiles_per_partition)
                } else {
                    Some(partitions_per_stage * tiles_per_partition)
                }
            }

            (Stage, Element) => {
                let tiles_per_stage = self.try_count_dim(Stage, Tile, dim)?;
                let elements_per_tile = self.try_count_dim(Tile, Element, dim)?;
                Some(tiles_per_stage * elements_per_tile)
            }

            (Partition, Element) => {
                let tiles_per_partition = self.try_count_dim(Partition, Tile, dim)?;
                let elements_per_tile = self.try_count_dim(Tile, Element, dim)?;
                Some(tiles_per_partition * elements_per_tile)
            }

            // Invalid transitions
            _ => None,
        }
    }

    pub fn count_total(&self, parent: TilingLevel, child: TilingLevel) -> u32 {
        self.try_count_total(parent, child)
            .unwrap_or_else(|| panic!("Invalid hierarchy: {parent:?} cannot contain {child:?}"))
    }

    pub fn try_count_total(
        &self,
        parent_level: TilingLevel,
        child_level: TilingLevel,
    ) -> Option<u32> {
        let m = self.try_count_dim(parent_level, child_level, MatmulDim::M)?;
        let n = self.try_count_dim(parent_level, child_level, MatmulDim::N)?;
        let k = self.try_count_dim(parent_level, child_level, MatmulDim::K)?;
        Some(m * n * k)
    }
}

macro_rules! define_2d_shape {
    ($name:ident) => {
        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        pub struct $name {
            pub m: u32,
            pub n: u32,
        }

        impl From<(u32, u32)> for $name {
            fn from(value: (u32, u32)) -> Self {
                Self {
                    m: value.0,
                    n: value.1,
                }
            }
        }

        impl $name {
            pub fn num_elems(&self) -> u32 {
                self.m * self.n
            }
        }
    };
}

macro_rules! define_3d_shape {
    ($name:ident) => {
        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        pub struct $name {
            pub m: u32,
            pub n: u32,
            pub k: u32,
        }

        impl From<(u32, u32, u32)> for $name {
            fn from(value: (u32, u32, u32)) -> Self {
                Self {
                    m: value.0,
                    n: value.1,
                    k: value.2,
                }
            }
        }

        impl $name {
            pub fn num_elems(&self) -> u32 {
                self.m * self.n * self.k
            }
        }
    };
}

/// Number of tiles in a stage partition
define_2d_shape!(TilesPerPartition);
/// Number of partitions in a stage
define_2d_shape!(PartitionsPerStage);
/// Number of elements in a tile
define_3d_shape!(TileShape);

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TilingScheme {
    pub tile_shape: TileShape,
    pub tiles_per_partition: TilesPerPartition,
    pub partitions_per_stage: PartitionsPerStage,
    pub stage_k_tile_count: u32,
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
    fn count_dim(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        dim: MatmulDim,
    ) -> u32 {
        self.try_count_dim(child_level, parent_level, dim)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    fn try_count_dim(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        dim: MatmulDim,
    ) -> Option<u32> {
        use MatmulDim::*;
        use TilingLevel::*;

        println!("{:?}", child_level);
        println!("{:?}", parent_level);

        match (child_level, parent_level) {
            (child, parent) if child == parent => Some(1),

            (Partition, Stage) => Some(match dim {
                M => self.partitions_per_stage.m,
                N => self.partitions_per_stage.n,
                K => 1,
            }),

            (Tile, Partition) => Some(match dim {
                M => self.tiles_per_partition.m,
                N => self.tiles_per_partition.n,
                K => 1,
            }),

            (Element, Tile) => Some(match dim {
                M => self.tile_shape.m,
                N => self.tile_shape.n,
                K => self.tile_shape.k,
            }),

            (Tile, Stage) => {
                let partitions_per_stage = self.try_count_dim(Partition, Stage, dim)?;
                let tiles_per_partition = self.try_count_dim(Tile, Partition, dim)?;

                // We must account for the k dim which is not considered in partitions
                if let K = dim {
                    Some(self.stage_k_tile_count * partitions_per_stage * tiles_per_partition)
                } else {
                    Some(partitions_per_stage * tiles_per_partition)
                }
            }

            (Element, Stage) => {
                let tiles_per_stage = self.try_count_dim(Tile, Stage, dim)?;
                let elements_per_tile = self.try_count_dim(Element, Tile, dim)?;
                Some(tiles_per_stage * elements_per_tile)
            }

            (Element, Partition) => {
                let tiles_per_partition = self.try_count_dim(Tile, Partition, dim)?;
                let elements_per_tile = self.try_count_dim(Element, Tile, dim)?;
                Some(tiles_per_partition * elements_per_tile)
            }

            // Invalid transitions
            _ => None,
        }
    }

    fn count_total(&self, child_level: TilingLevel, parent_level: TilingLevel) -> u32 {
        self.try_count_total(child_level, parent_level)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    fn try_count_total(&self, child_level: TilingLevel, parent_level: TilingLevel) -> Option<u32> {
        let m = self.try_count_dim(child_level, parent_level, MatmulDim::M)?;
        let n = self.try_count_dim(child_level, parent_level, MatmulDim::N)?;
        let k = self.try_count_dim(child_level, parent_level, MatmulDim::K)?;
        Some(m * n * k)
    }
}

macro_rules! tiling_dim_method {
    ($name:ident, $child:ident, $parent:ident, $dim:ident) => {
        pub fn $name(&self) -> u32 {
            self.count_dim(TilingLevel::$child, TilingLevel::$parent, MatmulDim::$dim)
        }
    };
}

macro_rules! tiling_total_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name(&self) -> u32 {
            self.count_total(TilingLevel::$child, TilingLevel::$parent)
        }
    };
}

impl TilingScheme {
    tiling_dim_method!(partitions_in_stage_m, Partition, Stage, M);
    tiling_dim_method!(partitions_in_stage_n, Partition, Stage, N);
    tiling_total_method!(partitions_in_stage_total, Partition, Stage);

    tiling_dim_method!(tiles_in_stage_m, Tile, Stage, M);
    tiling_dim_method!(tiles_in_stage_n, Tile, Stage, N);
    tiling_dim_method!(tiles_in_stage_k, Tile, Stage, K);
    tiling_total_method!(tiles_in_stage_total, Tile, Stage);

    tiling_dim_method!(elements_in_stage_m, Element, Stage, M);
    tiling_dim_method!(elements_in_stage_n, Element, Stage, N);
    tiling_dim_method!(elements_in_stage_k, Element, Stage, K);
    tiling_total_method!(elements_in_stage_total, Element, Stage);

    tiling_dim_method!(tiles_in_partition_m, Tile, Partition, M);
    tiling_dim_method!(tiles_in_partition_n, Tile, Partition, N);
    tiling_total_method!(tiles_in_partition_total, Tile, Partition);

    tiling_dim_method!(elements_in_partition_m, Element, Partition, M);
    tiling_dim_method!(elements_in_partition_n, Element, Partition, N);
    tiling_total_method!(elements_in_partition_total, Element, Partition);

    tiling_dim_method!(elements_in_tile_m, Element, Tile, M);
    tiling_dim_method!(elements_in_tile_n, Element, Tile, N);
    tiling_dim_method!(elements_in_tile_k, Element, Tile, K);
    tiling_total_method!(elements_in_tile_total, Element, Tile);
}

macro_rules! define_2d_shape {
    ($name:ident) => {
        #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
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

        impl From<$name> for (u32, u32) {
            fn from(value: $name) -> Self {
                (value.m, value.n)
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
        #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
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

        impl From<$name> for (u32, u32, u32) {
            fn from(value: $name) -> Self {
                (value.m, value.n, value.k)
            }
        }

        impl $name {
            pub fn num_elems(&self) -> u32 {
                self.m * self.n * self.k
            }
        }
    };
}

// Number of tiles in a stage partition
define_2d_shape!(TilesPerPartition);
// Number of partitions in a stage
define_2d_shape!(PartitionsPerStage);
// Number of elements in a tile
define_3d_shape!(TileShape);
// Shapes m,n,k of the problem
define_3d_shape!(MatmulProblemShape);

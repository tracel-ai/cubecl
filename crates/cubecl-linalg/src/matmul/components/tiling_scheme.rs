use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct TilingScheme {
    pub tile_size: TileSize,
    pub partition_size: PartitionSize,
    pub stage_size: StageSize,
}

impl TilingScheme {
    pub fn builder() -> TilingSchemeBuilder {
        TilingSchemeBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct TilingSchemeBuilder {
    tile_size: Option<TileSize>,
    partition_size: Option<PartitionSize>,
    stage_size: Option<StageSize>,
}

impl TilingSchemeBuilder {
    pub fn with_tile_size(mut self, tile_size: TileSize) -> Self {
        self.tile_size = Some(tile_size);
        self
    }

    pub fn with_partition_size(mut self, partition_size: PartitionSize) -> Self {
        self.partition_size = Some(partition_size);
        self
    }

    pub fn with_partitions_per_stage(mut self, stage_size: StageSize) -> Self {
        assert!(stage_size.k == 1, "Stage size k > 1 is not supported");
        self.stage_size = Some(stage_size);
        self
    }

    pub fn build(self) -> Result<TilingScheme, &'static str> {
        Ok(TilingScheme {
            tile_size: self.tile_size.ok_or("Missing tile_size")?,
            partition_size: self.partition_size.ok_or("Missing tiles_per_partition")?,
            stage_size: self.stage_size.ok_or("Missing partitions_per_stage")?,
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
        use TilingLevel::*;

        match (child_level, parent_level) {
            (child, parent) if child == parent => Some(1),

            (Partition, Stage) => Some(self.stage_size.get(dim)),

            (Tile, Partition) => Some(self.partition_size.get(dim)),

            (Element, Tile) => Some(self.tile_size.get(dim)),

            (Tile, Stage) => Some(
                self.try_count_dim(Partition, Stage, dim)?
                    * self.try_count_dim(Tile, Partition, dim)?,
            ),

            (Element, Stage) => Some(
                self.try_count_dim(Tile, Stage, dim)? * self.try_count_dim(Element, Tile, dim)?,
            ),

            (Element, Partition) => Some(
                self.try_count_dim(Tile, Partition, dim)?
                    * self.try_count_dim(Element, Tile, dim)?,
            ),

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
    tiling_dim_method!(partitions_in_stage_k, Partition, Stage, K);
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
    tiling_dim_method!(tiles_in_partition_k, Tile, Partition, K);
    tiling_total_method!(tiles_in_partition_total, Tile, Partition);

    tiling_dim_method!(elements_in_partition_m, Element, Partition, M);
    tiling_dim_method!(elements_in_partition_n, Element, Partition, N);
    tiling_dim_method!(elements_in_partition_k, Element, Partition, K);
    tiling_total_method!(elements_in_partition_total, Element, Partition);

    tiling_dim_method!(elements_in_tile_m, Element, Tile, M);
    tiling_dim_method!(elements_in_tile_n, Element, Tile, N);
    tiling_dim_method!(elements_in_tile_k, Element, Tile, K);
    tiling_total_method!(elements_in_tile_total, Element, Tile);
}

macro_rules! define_3d_size {
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
            pub fn get(&self, dim: MatmulDim) -> u32 {
                match dim {
                    MatmulDim::M => self.m,
                    MatmulDim::N => self.n,
                    MatmulDim::K => self.k,
                }
            }

            pub fn mn(&self) -> u32 {
                self.m * self.n
            }

            pub fn mk(&self) -> u32 {
                self.m * self.k
            }

            pub fn nk(&self) -> u32 {
                self.n * self.k
            }

            pub fn mnk(&self) -> u32 {
                self.m * self.n * self.k
            }
        }
    };
}

// Number of elements in a tile
define_3d_size!(TileSize);
// Number of tiles in a stage partition
define_3d_size!(PartitionSize);
// Number of partitions in a stage
define_3d_size!(StageSize);
// Shapes m,n,k of the problem
define_3d_size!(MatmulProblemSize);

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Ident;

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

    pub fn with_stage_size(mut self, stage_size: StageSize) -> Self {
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
    fn try_count_1d(
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
                self.try_count_1d(Partition, Stage, dim)?
                    * self.try_count_1d(Tile, Partition, dim)?,
            ),

            (Element, Stage) => {
                Some(self.try_count_1d(Tile, Stage, dim)? * self.try_count_1d(Element, Tile, dim)?)
            }

            (Element, Partition) => Some(
                self.try_count_1d(Tile, Partition, dim)? * self.try_count_1d(Element, Tile, dim)?,
            ),

            // Invalid transitions
            _ => None,
        }
    }

    fn try_count_2d(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        dim1: MatmulDim,
        dim2: MatmulDim,
    ) -> Option<u32> {
        Some(
            self.try_count_1d(child_level, parent_level, dim1)?
                * self.try_count_1d(child_level, parent_level, dim2)?,
        )
    }

    fn try_count_3d(&self, child_level: TilingLevel, parent_level: TilingLevel) -> Option<u32> {
        Some(
            self.try_count_1d(child_level, parent_level, MatmulDim::M)?
                * self.try_count_1d(child_level, parent_level, MatmulDim::N)?
                * self.try_count_1d(child_level, parent_level, MatmulDim::K)?,
        )
    }

    fn count_1d(&self, child_level: TilingLevel, parent_level: TilingLevel, dim: MatmulDim) -> u32 {
        self.try_count_1d(child_level, parent_level, dim)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    fn count_1d_ident_row<I: Into<Ident>>(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: I,
    ) -> u32 {
        match ident.into() {
            Ident::Lhs => self.count_1d(child_level, parent_level, MatmulDim::M),
            Ident::Rhs => self.count_1d(child_level, parent_level, MatmulDim::K),
            Ident::Out => self.count_1d(child_level, parent_level, MatmulDim::M),
        }
    }

    fn count_1d_ident_col<I: Into<Ident>>(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: I,
    ) -> u32 {
        match ident.into() {
            Ident::Lhs => self.count_1d(child_level, parent_level, MatmulDim::K),
            Ident::Rhs => self.count_1d(child_level, parent_level, MatmulDim::N),
            Ident::Out => self.count_1d(child_level, parent_level, MatmulDim::N),
        }
    }

    fn count_2d(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        dim1: MatmulDim,
        dim2: MatmulDim,
    ) -> u32 {
        self.try_count_2d(child_level, parent_level, dim1, dim2)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    fn count_2d_ident<I: Into<Ident>>(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: I,
    ) -> u32 {
        match ident.into() {
            Ident::Lhs => self.count_2d(child_level, parent_level, MatmulDim::M, MatmulDim::K),
            Ident::Rhs => self.count_2d(child_level, parent_level, MatmulDim::K, MatmulDim::N),
            Ident::Out => self.count_2d(child_level, parent_level, MatmulDim::M, MatmulDim::N),
        }
    }

    fn count_3d(&self, child_level: TilingLevel, parent_level: TilingLevel) -> u32 {
        self.try_count_3d(child_level, parent_level)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }
}

macro_rules! count_1d_method {
    ($name:ident, $child:ident, $parent:ident, $dim:ident) => {
        pub fn $name(&self) -> u32 {
            self.count_1d(TilingLevel::$child, TilingLevel::$parent, MatmulDim::$dim)
        }
    };
}

macro_rules! count_1d_ident_row_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.count_1d_ident_row(TilingLevel::$child, TilingLevel::$parent, ident)
        }
    };
}

macro_rules! count_1d_ident_col_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.count_1d_ident_col(TilingLevel::$child, TilingLevel::$parent, ident)
        }
    };
}

macro_rules! count_2d_method {
    ($name:ident, $child:ident, $parent:ident, $dim1:ident, $dim2:ident) => {
        pub fn $name(&self) -> u32 {
            self.count_2d(
                TilingLevel::$child,
                TilingLevel::$parent,
                MatmulDim::$dim1,
                MatmulDim::$dim2,
            )
        }
    };
}

macro_rules! count_2d_ident_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name<I: Into<Ident>>(&self, ident: I) -> u32 {
            self.count_2d_ident(TilingLevel::$child, TilingLevel::$parent, ident)
        }
    };
}

macro_rules! count_3d_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name(&self) -> u32 {
            self.count_3d(TilingLevel::$child, TilingLevel::$parent)
        }
    };
}

impl TilingScheme {
    count_1d_method!(partitions_in_stage_m, Partition, Stage, M);
    count_1d_method!(partitions_in_stage_n, Partition, Stage, N);
    count_1d_method!(partitions_in_stage_k, Partition, Stage, K);
    count_1d_ident_row_method!(partitions_in_stage_row, Partition, Stage);
    count_1d_ident_col_method!(partitions_in_stage_col, Partition, Stage);
    count_2d_method!(partitions_in_stage_mk, Partition, Stage, M, K);
    count_2d_method!(partitions_in_stage_nk, Partition, Stage, N, K);
    count_2d_method!(partitions_in_stage_mn, Partition, Stage, M, N);
    count_2d_ident_method!(partitions_in_stage, Partition, Stage);
    count_3d_method!(partitions_in_stage_mnk, Partition, Stage);

    count_1d_method!(tiles_in_stage_m, Tile, Stage, M);
    count_1d_method!(tiles_in_stage_n, Tile, Stage, N);
    count_1d_method!(tiles_in_stage_k, Tile, Stage, K);
    count_1d_ident_row_method!(tiles_in_stage_row, Tile, Stage);
    count_1d_ident_col_method!(tiles_in_stage_col, Tile, Stage);
    count_2d_method!(tiles_in_stage_mk, Tile, Stage, M, K);
    count_2d_method!(tiles_in_stage_nk, Tile, Stage, N, K);
    count_2d_method!(tiles_in_stage_mn, Tile, Stage, M, N);
    count_2d_ident_method!(tiles_in_stage, Tile, Stage);
    count_3d_method!(tiles_in_stage_mnk, Tile, Stage);

    count_1d_method!(elements_in_stage_m, Element, Stage, M);
    count_1d_method!(elements_in_stage_n, Element, Stage, N);
    count_1d_method!(elements_in_stage_k, Element, Stage, K);
    count_1d_ident_row_method!(elements_in_stage_row, Element, Stage);
    count_1d_ident_col_method!(elements_in_stage_col, Element, Stage);
    count_2d_method!(elements_in_stage_mk, Element, Stage, M, K);
    count_2d_method!(elements_in_stage_nk, Element, Stage, N, K);
    count_2d_method!(elements_in_stage_mn, Element, Stage, M, N);
    count_2d_ident_method!(elements_in_stage, Element, Stage);
    count_3d_method!(elements_in_stage_mnk, Element, Stage);

    count_1d_method!(tiles_in_partition_m, Tile, Partition, M);
    count_1d_method!(tiles_in_partition_n, Tile, Partition, N);
    count_1d_method!(tiles_in_partition_k, Tile, Partition, K);
    count_1d_ident_row_method!(tiles_in_partition_row, Tile, Partition);
    count_1d_ident_col_method!(tiles_in_partition_col, Tile, Partition);
    count_2d_method!(tiles_in_partition_mk, Tile, Partition, M, K);
    count_2d_method!(tiles_in_partition_nk, Tile, Partition, N, K);
    count_2d_method!(tiles_in_partition_mn, Tile, Partition, M, N);
    count_2d_ident_method!(tiles_in_partition, Tile, Partition);
    count_3d_method!(tiles_in_partition_mnk, Tile, Partition);

    count_1d_method!(elements_in_partition_m, Element, Partition, M);
    count_1d_method!(elements_in_partition_n, Element, Partition, N);
    count_1d_method!(elements_in_partition_k, Element, Partition, K);
    count_1d_ident_row_method!(elements_in_partition_row, Element, Partition);
    count_1d_ident_col_method!(elements_in_partition_col, Element, Partition);
    count_2d_method!(elements_in_partition_mk, Element, Partition, M, K);
    count_2d_method!(elements_in_partition_nk, Element, Partition, N, K);
    count_2d_method!(elements_in_partition_mn, Element, Partition, M, N);
    count_2d_ident_method!(elements_in_partition, Element, Partition);
    count_3d_method!(elements_in_partition_mnk, Element, Partition);

    count_1d_method!(elements_in_tile_m, Element, Tile, M);
    count_1d_method!(elements_in_tile_n, Element, Tile, N);
    count_1d_method!(elements_in_tile_k, Element, Tile, K);
    count_1d_ident_row_method!(elements_in_tile_row, Element, Tile);
    count_1d_ident_col_method!(elements_in_tile_col, Element, Tile);
    count_2d_method!(elements_in_tile_mk, Element, Tile, M, K);
    count_2d_method!(elements_in_tile_nk, Element, Tile, N, K);
    count_2d_method!(elements_in_tile_mn, Element, Tile, M, N);
    count_2d_ident_method!(elements_in_tile, Element, Tile);
    count_3d_method!(elements_in_tile_mnk, Element, Tile);
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

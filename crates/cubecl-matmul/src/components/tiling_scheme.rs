use super::StageIdent;
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

    fn count_1d(&self, child_level: TilingLevel, parent_level: TilingLevel, dim: MatmulDim) -> u32 {
        self.try_count_1d(child_level, parent_level, dim)
            .unwrap_or_else(|| {
                panic!("Invalid hierarchy: {parent_level:?} cannot contain {child_level:?}")
            })
    }

    fn count_1d_ident_row(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: StageIdent,
    ) -> u32 {
        match ident {
            StageIdent::Lhs => self.count_1d(child_level, parent_level, MatmulDim::M),
            StageIdent::Rhs => self.count_1d(child_level, parent_level, MatmulDim::K),
            StageIdent::Acc => self.count_1d(child_level, parent_level, MatmulDim::M),
            StageIdent::Out => self.count_1d(child_level, parent_level, MatmulDim::M),
        }
    }

    fn count_1d_ident_col(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: StageIdent,
    ) -> u32 {
        match ident {
            StageIdent::Lhs => self.count_1d(child_level, parent_level, MatmulDim::K),
            StageIdent::Rhs => self.count_1d(child_level, parent_level, MatmulDim::N),
            StageIdent::Acc => self.count_1d(child_level, parent_level, MatmulDim::N),
            StageIdent::Out => self.count_1d(child_level, parent_level, MatmulDim::N),
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

    fn count_2d_ident(
        &self,
        child_level: TilingLevel,
        parent_level: TilingLevel,
        ident: StageIdent,
    ) -> u32 {
        match ident {
            StageIdent::Lhs => self.count_2d(child_level, parent_level, MatmulDim::M, MatmulDim::K),
            StageIdent::Rhs => self.count_2d(child_level, parent_level, MatmulDim::K, MatmulDim::N),
            StageIdent::Acc => self.count_2d(child_level, parent_level, MatmulDim::M, MatmulDim::N),
            StageIdent::Out => self.count_2d(child_level, parent_level, MatmulDim::M, MatmulDim::N),
        }
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
        pub fn $name<I: Into<StageIdent>>(&self, ident: I) -> u32 {
            self.count_1d_ident_row(TilingLevel::$child, TilingLevel::$parent, ident.into())
        }
    };
}

macro_rules! count_1d_ident_col_method {
    ($name:ident, $child:ident, $parent:ident) => {
        pub fn $name<I: Into<StageIdent>>(&self, ident: I) -> u32 {
            self.count_1d_ident_col(TilingLevel::$child, TilingLevel::$parent, ident.into())
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
        pub fn $name<I: Into<StageIdent>>(&self, ident: I) -> u32 {
            self.count_2d_ident(TilingLevel::$child, TilingLevel::$parent, ident.into())
        }
    };
}

impl TilingScheme {
    count_1d_method!(stage_partitions_in_stage_m, StagePartition, Stage, M);
    count_1d_method!(stage_partitions_in_stage_n, StagePartition, Stage, N);
    count_1d_method!(stage_partitions_in_stage_k, StagePartition, Stage, K);
    count_1d_ident_row_method!(stage_partitions_in_stage_row, StagePartition, Stage);
    count_1d_ident_col_method!(stage_partitions_in_stage_col, StagePartition, Stage);
    count_2d_method!(stage_partitions_in_stage_mk, StagePartition, Stage, M, K);
    count_2d_method!(stage_partitions_in_stage_nk, StagePartition, Stage, N, K);
    count_2d_method!(stage_partitions_in_stage_mn, StagePartition, Stage, M, N);
    count_2d_ident_method!(stage_partitions_in_stage, StagePartition, Stage);

    count_1d_method!(tiles_in_stage_m, Tile, Stage, M);
    count_1d_method!(tiles_in_stage_n, Tile, Stage, N);
    count_1d_method!(tiles_in_stage_k, Tile, Stage, K);
    count_1d_ident_row_method!(tiles_in_stage_row, Tile, Stage);
    count_1d_ident_col_method!(tiles_in_stage_col, Tile, Stage);
    count_2d_method!(tiles_in_stage_mk, Tile, Stage, M, K);
    count_2d_method!(tiles_in_stage_nk, Tile, Stage, N, K);
    count_2d_method!(tiles_in_stage_mn, Tile, Stage, M, N);
    count_2d_ident_method!(tiles_in_stage, Tile, Stage);

    count_1d_method!(elements_in_stage_m, Element, Stage, M);
    count_1d_method!(elements_in_stage_n, Element, Stage, N);
    count_1d_method!(elements_in_stage_k, Element, Stage, K);
    count_1d_ident_row_method!(elements_in_stage_row, Element, Stage);
    count_1d_ident_col_method!(elements_in_stage_col, Element, Stage);
    count_2d_method!(elements_in_stage_mk, Element, Stage, M, K);
    count_2d_method!(elements_in_stage_nk, Element, Stage, N, K);
    count_2d_method!(elements_in_stage_mn, Element, Stage, M, N);
    count_2d_ident_method!(elements_in_stage, Element, Stage);

    count_1d_method!(tiles_in_stage_partition_m, Tile, StagePartition, M);
    count_1d_method!(tiles_in_stage_partition_n, Tile, StagePartition, N);
    count_1d_method!(tiles_in_stage_partition_k, Tile, StagePartition, K);
    count_1d_ident_row_method!(tiles_in_stage_partition_row, Tile, StagePartition);
    count_1d_ident_col_method!(tiles_in_stage_partition_col, Tile, StagePartition);
    count_2d_method!(tiles_in_stage_partition_mk, Tile, StagePartition, M, K);
    count_2d_method!(tiles_in_stage_partition_nk, Tile, StagePartition, N, K);
    count_2d_method!(tiles_in_stage_partition_mn, Tile, StagePartition, M, N);
    count_2d_ident_method!(tiles_in_stage_partition, Tile, StagePartition);

    count_1d_method!(elements_in_stage_partition_m, Element, StagePartition, M);
    count_1d_method!(elements_in_stage_partition_n, Element, StagePartition, N);
    count_1d_method!(elements_in_stage_partition_k, Element, StagePartition, K);
    count_1d_ident_row_method!(elements_in_stage_partition_row, Element, StagePartition);
    count_1d_ident_col_method!(elements_in_stage_partition_col, Element, StagePartition);
    count_2d_method!(
        elements_in_stage_partition_mk,
        Element,
        StagePartition,
        M,
        K
    );
    count_2d_method!(
        elements_in_stage_partition_nk,
        Element,
        StagePartition,
        N,
        K
    );
    count_2d_method!(
        elements_in_stage_partition_mn,
        Element,
        StagePartition,
        M,
        N
    );
    count_2d_ident_method!(elements_in_stage_partition, Element, StagePartition);

    count_1d_method!(elements_in_tile_m, Element, Tile, M);
    count_1d_method!(elements_in_tile_n, Element, Tile, N);
    count_1d_method!(elements_in_tile_k, Element, Tile, K);
    count_1d_ident_row_method!(elements_in_tile_row, Element, Tile);
    count_1d_ident_col_method!(elements_in_tile_col, Element, Tile);
    count_2d_method!(elements_in_tile_mk, Element, Tile, M, K);
    count_2d_method!(elements_in_tile_nk, Element, Tile, N, K);
    count_2d_method!(elements_in_tile_mn, Element, Tile, M, N);
    count_2d_ident_method!(elements_in_tile, Element, Tile);

    count_1d_method!(elements_in_global_partition_m, Element, GlobalPartition, M);
    count_1d_method!(elements_in_global_partition_n, Element, GlobalPartition, N);
    count_1d_method!(tiles_in_global_partition_m, Tile, GlobalPartition, M);
    count_1d_method!(tiles_in_global_partition_n, Tile, GlobalPartition, N);
    count_1d_method!(
        stage_partitions_in_global_partition_m,
        StagePartition,
        GlobalPartition,
        M
    );
    count_1d_method!(
        stage_partitions_in_global_partition_n,
        StagePartition,
        GlobalPartition,
        N
    );
    count_1d_method!(stages_in_global_partition_m, Stage, GlobalPartition, M);
    count_1d_method!(stages_in_global_partition_n, Stage, GlobalPartition, N);
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::AttentionPrecision;
use crate::components::attention_types::MSK;
use crate::components::fragment::{AttentionMatmul, AttentionMatmulConfig};
use crate::components::fragment::{
    FragmentLayout, FragmentLayoutExpand, FragmentMask, FragmentMaskExpand,
};
use cubecl_matmul::components::tile::StridedTile;

use cubecl_std::tensor::layout::Coordinates;

#[derive(CubeType)]
/// Mask tile for Tile Attention
/// It is an additive mask, which means the result of apply should be added, not multiplied
pub enum MaskTile<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    Materialized(MaterializedTileMask<AP, AM>),
    Logical(LogicalTileMask<AM::FragmentLayout>),
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaskTile<AP, AM> {
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] partition_pos: Coords2d,
        #[comptime] config: AM::Config,
    ) -> MaskTile<AP, AM> {
        let logical_mask = LogicalTileMask::<AM::FragmentLayout> {
            logical_iter_origin: LogicalIterOrigin::init(),
            partition_pos,
            causal: config.causal_mask(),
            out_of_bounds,
            fragment_layout: AM::softmax_layout(config),
        };

        if config.materialized_mask() {
            MaskTile::new_Materialized(MaterializedTileMask::<AP, AM> {
                fragment: AM::allocate_mask(config),
                logical_mask,
                config,
            })
        } else {
            MaskTile::new_Logical(logical_mask)
        }
    }

    /// Loads the mask data into the fragment, if a tile is given, otherwise only
    /// updates the logical mask
    pub fn update(&mut self, new_origin: Coords2d, tile: CubeOption<StridedTile<MSK<AP>>>) {
        match self {
            MaskTile::Materialized(materialized_tile_mask) => {
                // TODO read the tile
                materialized_tile_mask
                    .logical_mask
                    .update_origin(new_origin);

                materialized_tile_mask.update_tile(tile.unwrap())
            }
            MaskTile::Logical(logical_tile_mask) => logical_tile_mask.update_origin(new_origin),
        }
    }
}

#[derive(CubeType)]
pub struct LogicalIterOrigin {
    row: RuntimeCell<u32>,
    col: RuntimeCell<u32>,
}

#[cube]
impl LogicalIterOrigin {
    fn init() -> LogicalIterOrigin {
        LogicalIterOrigin {
            row: RuntimeCell::new(0),
            col: RuntimeCell::new(0),
        }
    }

    fn read(&self) -> Coords2d {
        (self.row.read(), self.col.read())
    }

    fn update(&mut self, new: Coords2d) {
        self.row.store(new.0);
        self.col.store(new.1);
    }
}

#[derive(CubeType)]
pub struct LogicalTileMask<F: FragmentLayout> {
    logical_iter_origin: LogicalIterOrigin,
    #[cube(comptime)]
    partition_pos: Coords2d,
    #[cube(comptime)]
    causal: bool,
    out_of_bounds: CubeOption<Coords2d>,
    fragment_layout: F,
}

#[cube]
impl<F: FragmentLayout> LogicalTileMask<F> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let pos_in_tile = self.fragment_layout.absolute_pos(local_pos);

        let pos = Coords2d::add(
            self.logical_iter_origin.read(),
            Coords2d::add(self.partition_pos.runtime(), pos_in_tile),
        );

        let causal_masked = self.causal && pos.0 < pos.1;

        let oob_masked = match self.out_of_bounds {
            CubeOption::Some(bounds) => !Coords2d::is_in_bounds(&pos, &bounds),
            CubeOption::None => false,
        };

        causal_masked || oob_masked
    }

    pub fn update_origin(&mut self, new_origin: Coords2d) {
        self.logical_iter_origin.update(new_origin);
    }
}

#[derive(CubeType)]
pub struct MaterializedTileMask<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    fragment: AM::Mask,
    logical_mask: LogicalTileMask<AM::FragmentLayout>,
    #[cube(comptime)]
    config: AM::Config,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaterializedTileMask<AP, AM> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let logical_masked = self.logical_mask.should_mask(local_pos);
        let materialized_masked = self.fragment.should_mask(local_pos);

        logical_masked || materialized_masked
    }

    pub fn update_tile(&mut self, tile: StridedTile<MSK<AP>>) {
        AM::fill_mask(&tile, &mut self.fragment, self.config);
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> FragmentMask for MaskTile<AP, AM> {
    fn should_mask(&self, local_pos: (u32, u32)) -> bool {
        match self {
            MaskTile::Materialized(materialized_tile_mask) => {
                materialized_tile_mask.should_mask(local_pos)
            }
            MaskTile::Logical(logical_tile_mask) => logical_tile_mask.should_mask(local_pos),
        }
    }
}

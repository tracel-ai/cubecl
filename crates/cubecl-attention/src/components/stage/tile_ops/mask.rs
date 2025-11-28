use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::AttentionPrecision;
use crate::components::attention_types::MSK;
use crate::components::tile::{
    FragmentLayout, FragmentLayoutExpand, FragmentMask, FragmentMaskExpand,
};
use crate::components::tile::{TileAttention, TileAttentionConfig};
use cubecl_matmul::components::tile::StridedTile;

use cubecl_std::tensor::layout::Coordinates;

#[derive(CubeType)]
/// Mask tile for Tile Attention
/// It is an additive mask, which means the result of apply should be added, not multiplied
pub enum MaskTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    /// When a mask tensor is supplied. Also contains a logical part
    Materialized(MaterializedTileMask<AP, TA>),
    /// When no mask tensor is supplied. Used for out of bounds and causal mask
    Logical(LogicalTileMask<TA::FragmentLayout>),
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> MaskTile<AP, TA> {
    pub fn new(
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] config: TA::Config,
    ) -> MaskTile<AP, TA> {
        let logical_mask = LogicalTileMask::<TA::FragmentLayout> {
            logical_iter_origin: LogicalIterOrigin::init(),
            causal: config.causal_mask(),
            out_of_bounds,
            fragment_layout: TA::softmax_layout(config),
        };

        if config.materialized_mask() {
            MaskTile::new_Materialized(MaterializedTileMask::<AP, TA> {
                fragment: TA::allocate_mask(config),
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
/// Gives the origin of the logical mask, which is updated when changing partition or tile within partition
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
    // Indicates where the logical mask currently starts
    logical_iter_origin: LogicalIterOrigin,
    #[cube(comptime)]
    // Whether to apply causal mask
    causal: bool,
    // Coordinates over which softmax is out of bounds, corresponds to seq_q, seq_kv of the problem
    out_of_bounds: CubeOption<Coords2d>,
    // Allows mapping local position of a unit to its absolute position
    fragment_layout: F,
}

#[cube]
impl<F: FragmentLayout> LogicalTileMask<F> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let pos_in_tile = self.fragment_layout.absolute_pos(local_pos);

        let pos = Coords2d::add(self.logical_iter_origin.read(), pos_in_tile);

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
pub struct MaterializedTileMask<AP: AttentionPrecision, TA: TileAttention<AP>> {
    fragment: TA::Mask,
    logical_mask: LogicalTileMask<TA::FragmentLayout>,
    #[cube(comptime)]
    config: TA::Config,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> MaterializedTileMask<AP, TA> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let logical_masked = self.logical_mask.should_mask(local_pos);
        let materialized_masked = self.fragment.should_mask(local_pos);

        logical_masked || materialized_masked
    }

    pub fn update_tile(&mut self, tile: StridedTile<MSK<AP>>) {
        TA::load_mask(&tile, &mut self.fragment, self.config);
    }
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> FragmentMask for MaskTile<AP, TA> {
    type Layout = <TA::Mask as FragmentMask>::Layout;

    fn should_mask(&self, local_pos: (u32, u32)) -> bool {
        match self {
            MaskTile::Materialized(materialized_tile_mask) => {
                materialized_tile_mask.should_mask(local_pos)
            }
            MaskTile::Logical(logical_tile_mask) => logical_tile_mask.should_mask(local_pos),
        }
    }
}

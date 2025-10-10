use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;
use cubecl_std::{CubeOption, CubeOptionExpand};

use crate::components::AttentionPrecision;
use crate::components::tile::dummy::AttentionMatmul;
use crate::components::tile::row::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::{FragmentLayout, FragmentLayoutExpand, MaskTile, MaskTileExpand};

use cubecl_std::tensor::layout::Coordinates;

#[derive(CubeType)]
pub struct LogicalTileMask<F: FragmentLayout> {
    origin: Coords2d,
    #[cube(comptime)]
    causal: bool,
    out_of_bounds: CubeOption<Coords2d>,
    fragment_layout: F,
}

#[cube]
impl<F: FragmentLayout> LogicalTileMask<F> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let pos_in_tile = self.fragment_layout.absolute_pos(local_pos);

        let pos = Coords2d::add(self.origin, pos_in_tile);

        let causal_masked = self.causal && pos.0 < pos.1;

        let oob_masked = match self.out_of_bounds {
            CubeOption::Some(bounds) => !Coords2d::is_in_bounds(&pos, &bounds),
            CubeOption::None => false,
        };

        causal_masked || oob_masked
    }
}

#[derive(CubeType)]
pub struct MaterializedTileMask<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    fragment: AM::Mask,
    logical_mask: LogicalTileMask<AM::FragmentLayout>,
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaterializedTileMask<AP, AM> {
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        let logical_masked = self.logical_mask.should_mask(local_pos);
        let materialized_masked = self.fragment.should_mask(local_pos);

        logical_masked || materialized_masked
    }
}

#[derive(CubeType)]
pub enum MaskFragment<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    Materialized(MaterializedTileMask<AP, AM>),
    Logical(LogicalTileMask<AM::FragmentLayout>),
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaskFragment<AP, AM> {
    pub fn new(
        origin: Coords2d,
        #[comptime] causal: bool,
        out_of_bounds: CubeOption<Coords2d>,
        #[comptime] materialized: bool,
        #[comptime] config: AM::Config,
    ) -> MaskFragment<AP, AM> {
        let logical_mask = LogicalTileMask::<AM::FragmentLayout> {
            origin,
            causal,
            out_of_bounds,
            fragment_layout: AM::softmax_layout(config),
        };

        if materialized {
            MaskFragment::new_Materialized(MaterializedTileMask::<AP, AM> {
                fragment: AM::allocate_mask(config),
                logical_mask,
            })
        } else {
            MaskFragment::new_Logical(logical_mask)
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaskTile for MaskFragment<AP, AM> {
    type Fragment = AM::Mask;

    fn apply<E: Float>(this: &Self, local_pos: Coords2d) -> E {
        let should_mask = match this {
            MaskFragment::Materialized(materialized_tile_mask) => {
                materialized_tile_mask.should_mask(local_pos)
            }
            MaskFragment::Logical(logical_tile_mask) => logical_tile_mask.should_mask(local_pos),
        };

        E::cast_from(should_mask) * E::min_value()
    }

    fn fragment_mut(&mut self) -> &mut Self::Fragment {
        match self {
            MaskFragment::Materialized(materialized_tile_mask) => {
                &mut materialized_tile_mask.fragment
            }
            MaskFragment::Logical(_) => {
                panic!("Tried to get fragment of logical mask")
            }
        }
    }
}

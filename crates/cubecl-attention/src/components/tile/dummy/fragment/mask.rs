use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::tile::{MaskTile, MaskTileExpand};
use crate::components::tile::dummy::AttentionMatmul;
use cubecl_std::tensor::layout::Coordinates;

#[derive(CubeType)]
pub struct LogicalTileMask {
    origin: Coords2d,
    #[cube(comptime)]
    causal: bool,
    out_of_bounds: CubeOption<Coords2d>,
}

impl LogicalTileMask {
    pub fn should_mask(&self, pos_in_tile: Coords2d) -> bool {
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
    logical_mask: LogicalTileMask,
}

#[derive(CubeType)]
pub enum MaskFragment<AP: AttentionPrecision, AM: AttentionMatmul<AP>> {
    Materialized(MaterializedTileMask<AP, AM>),
    Logical(LogicalTileMask),
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
        let logical_mask = LogicalTileMask {
            origin,
            causal,
            out_of_bounds,
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
impl<AP: AttentionPrecision, AM: AttentionMatmul<AP>> MaskTile<AP> for MaskFragment<AP, AM> {
    type Fragment = AM::Mask;

    fn fragment(&self) -> &Self::Fragment {
        todo!()
    }
}

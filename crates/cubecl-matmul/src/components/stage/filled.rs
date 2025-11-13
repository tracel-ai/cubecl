use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::{
    stage::{Stage, StageFamily, TilingLayout},
    tile::io::Filled,
};

pub struct FilledStageFamily;

impl StageFamily for FilledStageFamily {
    type TileKind = Filled;

    type Stage<ES: Numeric, T: TilingLayout> = FilledStage<ES>;
}

#[derive(CubeType, Clone)]
pub struct FilledStage<ES: Numeric> {
    value: ES,
}

#[cube]
impl<ES: Numeric> FilledStage<ES> {
    pub fn new(value: ES) -> Self {
        FilledStage::<ES> { value }
    }
}

#[cube]
impl<ES: Numeric> Stage<ES, ReadOnly> for FilledStage<ES> {
    type TileKind = Filled;

    fn tile(this: &Self, _tile: Coords2d) -> ES {
        this.value
    }
}

use std::marker::PhantomData;

use crate::matmul::components::TensorIdent;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::tile::Tile;
use crate::matmul::components::tile::TileConfig;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Stage reader for LHS
pub struct Reader<I: TensorIdent, ES: Numeric, T: TilingLayout> {
    pub stage: Stage<ES, T>,
    #[cube(comptime)]
    _ident: PhantomData<I>,
}

pub struct StageReaderFamily<I: TensorIdent> {
    _ident: PhantomData<I>,
}

impl<I: TensorIdent> ReaderFamily for StageReaderFamily<I> {
    type Reader<ES: Numeric, T: TilingLayout> = Reader<I, ES, T>;
}

#[cube]
impl<I: TensorIdent, ES: Numeric, T: TilingLayout> Reader<I, ES, T> {
    pub fn read_tile<TC: TileConfig>(
        this: &Self,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: CommonStageConfig<TC>,
    ) -> Tile<ES> {
        this.stage
            .get_tile::<CommonStageConfig<TC>>(x_offset, y_offset, I::IDENT, config)
    }
}

#[cube]
impl<I: TensorIdent, ES: Numeric, T: TilingLayout> Reader<I, ES, T> {
    pub fn new(stage: Stage<ES, T>) -> Reader<I, ES, T> {
        Reader::<I, ES, T> {
            stage,
            _ident: PhantomData::<I>,
        }
    }
}

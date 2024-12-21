use crate::matmul::components::stage::multi_buffer;
use crate::matmul::components::stage::ReaderFamily;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::tile::TileConfig;
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
/// Stage reader for LHS
pub struct LhsReader<ES: Numeric> {
    pub stage: Stage<ES>,
}

#[derive(CubeType)]
/// Stage reader for RHS
pub struct RhsReader<ES: Numeric> {
    pub stage: Stage<ES>,
}

pub struct LhsReaderFamily;
pub struct RhsReaderFamily;

impl ReaderFamily for LhsReaderFamily {
    type Reader<I: Numeric> = LhsReader<I>;
}

impl ReaderFamily for RhsReaderFamily {
    type Reader<I: Numeric> = RhsReader<I>;
}

#[cube]
impl<ES: Numeric> LhsReader<ES> {
    pub fn read_tile<T: TileConfig>(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        #[comptime] config: multi_buffer::Config<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<multi_buffer::Config<T>>(
            compute_plane_offset,
            buffer_offset,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric> RhsReader<ES> {
    pub fn read_tile<T: TileConfig>(
        this: &Self,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: multi_buffer::Config<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<multi_buffer::Config<T>>(
            buffer_offset,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric> LhsReader<ES> {
    pub fn new(stage: Stage<ES>) -> LhsReader<ES> {
        LhsReader::<ES> { stage }
    }
}

#[cube]
impl<ES: Numeric> RhsReader<ES> {
    pub fn new(stage: Stage<ES>) -> RhsReader<ES> {
        RhsReader::<ES> { stage }
    }
}

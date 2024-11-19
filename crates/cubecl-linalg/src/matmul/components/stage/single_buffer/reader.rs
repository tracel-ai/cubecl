use crate::matmul::components::stage::single_buffer;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{stage::Stage, tile, Ident};

#[derive(CubeType)]
pub struct LhsBufferReader<ES: Numeric> {
    pub stage: Stage<ES>,
    pub buffer: u32,
}

#[derive(CubeType)]
pub struct RhsBufferReader<ES: Numeric> {
    pub stage: Stage<ES>,
    pub buffer: u32,
}

#[cube]
impl<ES: Numeric> LhsBufferReader<ES> {
    pub fn read_tile<T: tile::Config>(
        this: &Self,
        compute_plane_offset: u32,
        #[comptime] config: single_buffer::Config<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<single_buffer::Config<T>>(
            compute_plane_offset,
            this.buffer,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric> RhsBufferReader<ES> {
    pub fn read_tile<T: tile::Config>(
        this: &Self,
        accumulator_offset: u32,
        #[comptime] config: single_buffer::Config<T>,
    ) -> Slice<Line<ES>> {
        this.stage.get_tile::<single_buffer::Config<T>>(
            this.buffer,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}

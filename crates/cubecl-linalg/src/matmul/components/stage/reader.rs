use crate::matmul::components::stage::StageReader;
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::staging::Stage;
use super::Config;

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

#[cube]
impl<ES: Numeric> StageReader<ES> for LhsReader<ES> {
    fn read_tile<S: Config>(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
        #[comptime] config: S,
    ) -> Slice<Line<ES>> {
        this.stage
            .get_tile::<S>(compute_plane_offset, buffer_offset, Ident::Lhs, config)
    }
}

#[cube]
impl<ES: Numeric> StageReader<ES> for RhsReader<ES> {
    fn read_tile<S: Config>(
        this: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> Slice<Line<ES>> {
        this.stage
            .get_tile::<S>(buffer_offset, accumulator_offset, Ident::Rhs, config)
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

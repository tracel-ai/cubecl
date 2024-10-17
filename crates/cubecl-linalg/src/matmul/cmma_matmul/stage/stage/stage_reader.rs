use crate::matmul::cmma_matmul::stage::CmmaStageMatmulConfig;
use crate::matmul::matmul_stage::{Stage, StageReader};
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::Stage;

#[derive(CubeType)]
pub struct LhsStageReader<ES: Numeric> {
    pub stage: Stage,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsStageReader<ES: Numeric> {
    pub stage: Stage,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<ES: Numeric> StageReader<ES> for LhsStageReader<ES> {
    fn read_tile<T: TmmConfig>(
        self_: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
        #[comptime] config: CmmaStageMatmulConfig<T>,
    ) -> &Slice<'_, Line<ES>> {
        S::get_tile(
            &self_.stage,
            compute_plane_offset,
            buffer_offset,
            Ident::Lhs,
            config,
        )
    }
}

#[cube]
impl<ES: Numeric, S: Stage<ES>> StageReader<ES> for RhsStageReader<ES, S> {
    type Config = S::Config;

    fn read_tile(
        self_: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: Self::Config,
    ) -> &Slice<'_, Line<ES>> {
        S::get_tile(
            &self_.stage,
            buffer_offset,
            accumulator_offset,
            Ident::Rhs,
            config,
        )
    }
}

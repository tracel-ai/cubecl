use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::matmul_stage::Stage;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::SmmConfig;

#[cube]
pub trait StageReader<ES: Numeric>: CubeType {
    type Config: SmmConfig;

    fn read_tile(
        self_: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: Self::Config,
    ) -> &Slice<'_, Line<ES>>;
}

#[derive(CubeType)]
pub struct LhsStageReader<ES: Numeric, S: Stage<ES>> {
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[derive(CubeType)]
pub struct RhsStageReader<ES: Numeric, S: Stage<ES>> {
    pub stage: S,
    pub _e: PhantomData<ES>,
}

#[cube]
impl<ES: Numeric, S: Stage<ES, Config = CmmaConfig>> StageReader<ES> for LhsStageReader<ES, S> {
    type Config = CmmaConfig;

    fn read_tile(
        self_: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
        #[comptime] config: Self::Config,
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
impl<ES: Numeric, S: Stage<ES, Config = CmmaConfig>> StageReader<ES> for RhsStageReader<ES, S> {
    type Config = CmmaConfig;

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

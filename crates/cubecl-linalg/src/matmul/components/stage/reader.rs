use std::marker::PhantomData;

use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage::StageReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::stage::Stage;
use super::SmmConfig;

#[derive(CubeType)]
/// Stage reader for LHS
pub struct LhsStageReader<ES: Numeric, S: SmmConfig> {
    pub stage: Stage<ES>,
    pub _config: PhantomData<S>,
}

#[derive(CubeType)]
/// Stage reader for RHS
pub struct RhsStageReader<ES: Numeric, S: SmmConfig> {
    pub stage: Stage<ES>,
    pub _config: PhantomData<S>,
}

#[cube]
impl<ES: Numeric, S: SmmConfig> StageReader<ES, S> for LhsStageReader<ES, S> {
    fn read_tile(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>> {
        this.stage
            .get_tile::<S>(compute_plane_offset, buffer_offset, Ident::Lhs, config)
    }
}

#[cube]
impl<ES: Numeric, S: SmmConfig> StageReader<ES, S> for RhsStageReader<ES, S> {
    fn read_tile(
        this: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>> {
        this.stage
            .get_tile::<S>(buffer_offset, accumulator_offset, Ident::Rhs, config)
    }
}

#[cube]
impl<ES: Numeric, S: SmmConfig> LhsStageReader<ES, S> {
    pub fn new(stage: Stage<ES>) -> LhsStageReader<ES, S> {
        LhsStageReader::<ES, S> {
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<ES: Numeric, S: SmmConfig> RhsStageReader<ES, S> {
    pub fn new(stage: Stage<ES>) -> RhsStageReader<ES, S> {
        RhsStageReader::<ES, S> {
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

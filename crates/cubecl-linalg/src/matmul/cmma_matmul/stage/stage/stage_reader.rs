use std::marker::PhantomData;

use crate::matmul::matmul_stage::{SmmConfig, StageReader};
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Stage;

#[derive(CubeType)]
pub struct LhsStageReader<ES: Numeric, S: SmmConfig> {
    pub stage: Stage<ES>,
    pub _config: PhantomData<S>,
}

#[derive(CubeType)]
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

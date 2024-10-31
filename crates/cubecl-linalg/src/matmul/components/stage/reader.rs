use std::marker::PhantomData;

use crate::matmul::components::Ident;
use crate::matmul::components::stage::StageReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::stage::Stage;
use super::Config;

#[derive(CubeType)]
/// Stage reader for LHS
pub struct LhsReader<ES: Numeric, S: Config> {
    pub stage: Stage<ES>,
    pub _config: PhantomData<S>,
}

#[derive(CubeType)]
/// Stage reader for RHS
pub struct RhsReader<ES: Numeric, S: Config> {
    pub stage: Stage<ES>,
    pub _config: PhantomData<S>,
}

#[cube]
impl<ES: Numeric, S: Config> StageReader<ES, S> for LhsReader<ES, S> {
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
impl<ES: Numeric, S: Config> StageReader<ES, S> for RhsReader<ES, S> {
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
impl<ES: Numeric, S: Config> LhsReader<ES, S> {
    pub fn new(stage: Stage<ES>) -> LhsReader<ES, S> {
        LhsReader::<ES, S> {
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

#[cube]
impl<ES: Numeric, S: Config> RhsReader<ES, S> {
    pub fn new(stage: Stage<ES>) -> RhsReader<ES, S> {
        RhsReader::<ES, S> {
            stage,
            _config: PhantomData::<S>.runtime(),
        }
    }
}

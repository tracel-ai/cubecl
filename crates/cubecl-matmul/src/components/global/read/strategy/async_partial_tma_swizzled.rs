use crate::components::global::GlobalConfig;
use crate::components::stage::{SwizzledStage, SwizzledStageFamily};
use crate::components::{InvalidConfigError, MatmulIdent, MatrixPrecision, TilingScheme};
use crate::components::{
    MatrixLayout,
    global::read::{PartialLoadingStrategy, async_tma::AsyncTma},
};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use crate::components::{
    global::{RoleRule, multi_stage::LoadMaxRoundPlaneCount},
    stage::StridedTilingLayout,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::Barrier};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using TMA load instructions.
/// Uses strided tiling, with swizzling to take care of bank conflicts.
pub struct AsyncPartialTmaSwizzledLoading {}

impl LoadingValidation for AsyncPartialTmaSwizzledLoading {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        StridedTilingLayout::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for AsyncPartialTmaSwizzledLoading {
    fn max_round_plane_count(
        _tiling_scheme: &TilingScheme,
        _ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        4
    }
}

#[cube]
impl PartialLoadingStrategy for AsyncPartialTmaSwizzledLoading {
    type TilingLayout = StridedTilingLayout;
    type SyncStrategy = AsyncTma;
    type Stage = SwizzledStageFamily;

    type Job<IP: MatrixPrecision> = AsyncPartialTmaSwizzledJob;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] _line_size: u32,
        #[comptime] config: G,
    ) -> Self::Job<IP> {
        let role_rule_config = config.role_rule_config();
        let is_elected = RoleRule::new(role_rule_config).elect_load_leader();

        AsyncPartialTmaSwizzledJob {
            is_elected,
            stage_index,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncPartialTmaSwizzledJob {
    is_elected: bool,

    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
}

#[cube]
impl<IP: MatrixPrecision> LoadingJob<IP, StridedTilingLayout, AsyncTma>
    for AsyncPartialTmaSwizzledJob
{
    type Stage = SwizzledStageFamily;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] _task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut SwizzledStage<IP::Stage>,
        barrier: &mut Barrier,
        #[comptime] config: G,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        if this.is_elected {
            let config = comptime![config.stage_memory_config(this.ident)];

            let (offs_row, offs_col) = comptime![match this.ident {
                MatmulIdent::Lhs => (0, this.stage_index * config.elements_in_stage_col()),
                MatmulIdent::Rhs => (this.stage_index * config.elements_in_stage_row(), 0),
                _ => (0, 0),
            }]
            .runtime();

            let global_view = global_iter.view();
            let stage = stage.as_slice_mut(1u32);

            let pos = match config.matrix_layout {
                MatrixLayout::RowMajor => (offs_row, offs_col),
                MatrixLayout::ColMajor => (offs_row, offs_col),
            };

            global_view.tensor_map_load(barrier, &mut stage.try_cast_unchecked(), pos);
        }
    }

    fn task_count(_this: &Self) -> comptime_type!(u32) {
        1u32
    }
}

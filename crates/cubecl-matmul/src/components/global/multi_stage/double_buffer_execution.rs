use crate::components::global::LoadingSides;
use crate::components::global::RoleRule;
use crate::components::global::Specializer;
use crate::components::global::SpecializerKind;
use crate::components::global::multi_stage::DoubleBufferingEventListener;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::read::StageBuffer;
use crate::components::global::{GlobalConfig, GlobalWriter};
use crate::components::stage::PartitionScheduler;
use crate::components::{MatmulPrecision, stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// Read the first stage for both Lhs and Rhs
///
/// If there is specialization, will add a runtime if to determine the role of the plane
pub fn read_first<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LJ: JobExecutor<G>,
    RJ: JobExecutor<G>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_global_reader: &mut LJ,
    rhs_global_reader: &mut RJ,
    specializer: &Specializer,
    #[comptime] stage_to_load: StageBuffer,
    #[comptime] config: G,
) {
    match comptime!(specializer.kind) {
        SpecializerKind::Specialized {
            main_flow_loading_side,
            load_only_loading_side,
            role_rule_config,
        } => {
            let rule = RoleRule::new(role_rule_config);
            if !rule.is_load_only() {
                if main_flow_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_global_reader, stage_to_load, config);
                }
                if main_flow_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_global_reader, stage_to_load, config);
                }
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_global_reader, stage_to_load, config);
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_global_reader, stage_to_load, config);
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            LJ::execute_whole_job(lhs_global_reader, stage_to_load, config);
            RJ::execute_whole_job(rhs_global_reader, stage_to_load, config);
        }
    };
}

#[cube]
/// Execute on the current stage while loading the next stage
///
/// If there is specialization, will add a runtime if to determine the role of the plane
pub fn execute_current_and_read_next<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LJ: JobExecutor<G>,
    RJ: JobExecutor<G>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_stage: &SMM::LhsStage,
    rhs_stage: &SMM::RhsStage,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulators,
    lhs_global_reader: &mut LJ,
    rhs_global_reader: &mut RJ,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] stage_to_load: StageBuffer,
    #[comptime] config: G,
) {
    match comptime!(specializer.kind) {
        SpecializerKind::Specialized {
            main_flow_loading_side,
            load_only_loading_side,
            role_rule_config,
        } => {
            let rule = RoleRule::new(role_rule_config);
            if !rule.is_load_only() {
                SMM::execute_with_listener::<DoubleBufferingEventListener<LJ, RJ, G>>(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config(),
                    DoubleBufferingEventListener::new(
                        stage_to_load,
                        lhs_global_reader,
                        rhs_global_reader,
                        config,
                        main_flow_loading_side,
                    ),
                    partition_scheduler,
                );
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_global_reader, stage_to_load, config);
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_global_reader, stage_to_load, config);
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            SMM::execute_with_listener::<DoubleBufferingEventListener<LJ, RJ, G>>(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config(),
                DoubleBufferingEventListener::new(
                    stage_to_load,
                    lhs_global_reader,
                    rhs_global_reader,
                    config,
                    LoadingSides::Both,
                ),
                partition_scheduler,
            );
        }
    };
}

#[cube]
/// Execute on the last stage, then write results
///
/// If there is specialization, will add a runtime if to determine the role of the plane
pub fn execute_last_and_write_results<
    MP: MatmulPrecision,
    GW: GlobalWriter<MP::Acc>,
    SMM: stage::StageMatmul<MP, OutStage = GW::Stage>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_stage: &SMM::LhsStage,
    rhs_stage: &SMM::RhsStage,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulators,
    out_writer: &mut GW,
    specializer: &Specializer,
    partition_scheduler: &PartitionScheduler,
    #[comptime] config: G,
) {
    let mut out_stage = GW::stage(out_writer);

    match comptime!(specializer.kind) {
        SpecializerKind::Specialized {
            main_flow_loading_side: _,
            load_only_loading_side: _,
            role_rule_config,
        } => {
            let rule = RoleRule::new(role_rule_config);
            if !rule.is_load_only() {
                SMM::execute(
                    lhs_stage,
                    rhs_stage,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config(),
                    partition_scheduler,
                );

                SMM::write_results::<GW, G>(
                    acc,
                    &mut out_stage,
                    out_writer,
                    partition_scheduler,
                    config.stage_config(),
                    config,
                );
            }
        }
        SpecializerKind::NotSpecialized => {
            SMM::execute(
                lhs_stage,
                rhs_stage,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config(),
                partition_scheduler,
            );

            SMM::write_results::<GW, G>(
                acc,
                &mut out_stage,
                out_writer,
                partition_scheduler,
                config.stage_config(),
                config,
            );
        }
    }
}

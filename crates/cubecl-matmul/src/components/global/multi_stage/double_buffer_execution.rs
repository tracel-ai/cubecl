use crate::components::global::GlobalConfig;
use crate::components::global::LoadingSides;
use crate::components::global::RoleRule;
use crate::components::global::Specializer;
use crate::components::global::SpecializerKind;
use crate::components::global::load::BufferId;
use crate::components::global::multi_stage::DoubleBufferingEventListener;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::{MatmulPrecision, stage};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn load_first<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LJ: JobExecutor<G>,
    RJ: JobExecutor<G>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_loader: &mut LJ,
    rhs_loader: &mut RJ,
    specializer: &Specializer,
    #[comptime] buffer_to_load: BufferId,
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
                    LJ::execute_whole_job(lhs_loader, buffer_to_load, config);
                }
                if main_flow_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_loader, buffer_to_load, config);
                }
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_loader, buffer_to_load, config);
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_loader, buffer_to_load, config);
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            LJ::execute_whole_job(lhs_loader, buffer_to_load, config);
            RJ::execute_whole_job(rhs_loader, buffer_to_load, config);
        }
    };
}

#[cube]
pub fn execute_current_and_load_next<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LJ: JobExecutor<G>,
    RJ: JobExecutor<G>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    lhs_loader: &mut LJ,
    rhs_loader: &mut RJ,
    specializer: &Specializer,
    #[comptime] buffer_to_load: BufferId,
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
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config(),
                    DoubleBufferingEventListener::new(
                        buffer_to_load,
                        lhs_loader,
                        rhs_loader,
                        config,
                        main_flow_loading_side,
                    ),
                );
            } else {
                if load_only_loading_side.includes_lhs() {
                    LJ::execute_whole_job(lhs_loader, buffer_to_load, config);
                }
                if load_only_loading_side.includes_rhs() {
                    RJ::execute_whole_job(rhs_loader, buffer_to_load, config);
                }
            }
        }
        SpecializerKind::NotSpecialized => {
            SMM::execute_with_listener::<DoubleBufferingEventListener<LJ, RJ, G>>(
                lhs_reader,
                rhs_reader,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config(),
                DoubleBufferingEventListener::new(
                    buffer_to_load,
                    lhs_loader,
                    rhs_loader,
                    config,
                    LoadingSides::Both,
                ),
            );
        }
    };
}

#[cube]
pub fn execute_last_and_write_results<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    G: GlobalConfig<StageConfig = SMM::Config>,
>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    out_writer: &mut SMM::Writer,
    specializer: &Specializer,
    #[comptime] config: G,
) {
    match comptime!(specializer.kind) {
        SpecializerKind::Specialized {
            main_flow_loading_side: _,
            load_only_loading_side: _,
            role_rule_config,
        } => {
            let rule = RoleRule::new(role_rule_config);
            if !rule.is_load_only() {
                SMM::execute(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    config.stage_config(),
                );

                SMM::write_results::<G>(acc, out_writer, config.stage_config(), config);
            }
        }
        SpecializerKind::NotSpecialized => {
            SMM::execute(
                lhs_reader,
                rhs_reader,
                lhs_tile,
                rhs_tile,
                acc,
                config.stage_config(),
            );

            SMM::write_results::<G>(acc, out_writer, config.stage_config(), config);
        }
    }
}

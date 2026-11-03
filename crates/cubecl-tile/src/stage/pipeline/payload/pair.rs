//! The two-operand payload: what a contraction's slots hold.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

/// Two operands staged together; `Clone` duplicates handles, not buffers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct OperandPair {
    pub lhs: Tile,
    pub rhs: Tile,
}

#[cube]
impl Payload<OperandPair> for OperandPair {
    fn operands(&self) -> comptime_type!(Vec<StageOperand>) {
        let lhs = self.lhs.stage_operand();
        let rhs = self.rhs.stage_operand();
        // The scales a stage keeps beside its values come after both operands, so each operand
        // keeps its place in the slot's refills.
        let lhs_scales = self.lhs.scale_operands();
        let rhs_scales = self.rhs.scale_operands();
        comptime!(
            [lhs, rhs]
                .into_iter()
                .chain(lhs_scales)
                .chain(rhs_scales)
                .collect()
        )
    }

    fn staged(
        &self,
        first: &OperandPair,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> OperandPair {
        let lhs = if comptime!(refills[0] == Refill::Shared) {
            first.lhs.clone()
        } else {
            stage_one(&self.lhs, comptime!(spec.clone()))
        };
        let rhs = if comptime!(refills[1] == Refill::Shared) {
            first.rhs.clone()
        } else {
            stage_one(&self.rhs, comptime!(spec))
        };
        OperandPair { lhs, rhs }
    }

    fn free(&self) {
        self.lhs.free_stage();
        self.rhs.free_stage();
    }

    fn inside_runs(&self) -> OperandPair {
        OperandPair {
            lhs: self.lhs.inside_run(),
            rhs: self.rhs.inside_run(),
        }
    }

    fn bring(
        &mut self,
        src: &OperandPair,
        meeting: &Meeting,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
        #[comptime] only: Refill,
    ) {
        if comptime!(refills[0] == only) {
            meeting.fill(&mut self.lhs, &src.lhs.at(region));
        }
        if comptime!(refills[1] == only) {
            meeting.fill(&mut self.rhs, &src.rhs.at(region));
        }
    }
}

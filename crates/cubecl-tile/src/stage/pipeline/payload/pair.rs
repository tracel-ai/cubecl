//! The two-operand payload: what a contraction's slots hold.

use cubecl::prelude::*;

use crate::*;

use super::base::{Payload, PayloadExpand, StageOperand, StageSpec, stage_one};

/// Which operands of a contraction its slots stage. One not staged is read where it lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StagedOperands {
    Both,
    Lhs,
    Rhs,
}

impl StagedOperands {
    /// Whether the lhs is staged.
    pub fn lhs(self) -> bool {
        match self {
            Self::Both | Self::Lhs => true,
            Self::Rhs => false,
        }
    }

    /// Whether the rhs is staged.
    pub fn rhs(self) -> bool {
        match self {
            Self::Both | Self::Rhs => true,
            Self::Lhs => false,
        }
    }
}

/// Two operands a walk takes together, each staged or read where it lies; `Clone` duplicates
/// handles, not buffers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct OperandPair {
    pub lhs: Tile,
    pub rhs: Tile,
    #[cube(comptime)]
    pub staged: StagedOperands,
}

/// `operand` in a slot whose refill of it is `refill`: a stage of its own, the one `first` slot
/// holds, or the operand itself, read where it lies.
#[cube]
fn staged_operand(
    operand: &Tile,
    first: &Tile,
    #[comptime] spec: StageSpec,
    #[comptime] refill: Refill,
) -> Tile {
    match refill {
        Refill::EveryRegion | Refill::Once => stage_one(operand, spec),
        Refill::Shared => first.clone(),
        Refill::InPlace => operand.clone(),
    }
}

/// `held` as its reader takes it at `region`, `src` the operand it was made from.
#[cube]
fn operand_at(held: &Tile, src: &Tile, region: &Region, #[comptime] refill: Refill) -> Tile {
    match refill {
        Refill::EveryRegion | Refill::Once | Refill::Shared => held.clone(),
        Refill::InPlace => src.at(region),
    }
}

/// Free `operand`'s stage where it is `staged`.
#[cube]
fn free_if_staged(operand: &Tile, #[comptime] staged: bool) {
    if staged {
        operand.free_stage();
    }
}

#[cube]
impl Payload<OperandPair> for OperandPair {
    fn operands(&self) -> comptime_type!(Vec<Option<StageOperand>>) {
        let lhs = self.lhs.stage_operand();
        let rhs = self.rhs.stage_operand();
        // The scales a stage keeps beside its values come after both operands, so each operand
        // keeps its place in the slot's refills.
        let lhs_scales = self.lhs.scale_operands();
        let rhs_scales = self.rhs.scale_operands();
        comptime!({
            let values = [
                self.staged.lhs().then_some(lhs),
                self.staged.rhs().then_some(rhs),
            ];
            let lhs_scales = lhs_scales.into_iter().filter(|_| self.staged.lhs());
            let rhs_scales = rhs_scales.into_iter().filter(|_| self.staged.rhs());
            let scales = lhs_scales.chain(rhs_scales).map(Some);
            values.into_iter().chain(scales).collect()
        })
    }

    fn staged(
        &self,
        first: &OperandPair,
        #[comptime] spec: StageSpec,
        #[comptime] refills: Vec<Refill>,
    ) -> OperandPair {
        OperandPair {
            lhs: staged_operand(&self.lhs, &first.lhs, spec.clone(), refills[0]),
            rhs: staged_operand(&self.rhs, &first.rhs, comptime!(spec), refills[1]),
            staged: comptime!(self.staged),
        }
    }

    fn free(&self) {
        free_if_staged(&self.lhs, comptime!(self.staged.lhs()));
        free_if_staged(&self.rhs, comptime!(self.staged.rhs()));
    }

    fn placed_at(
        &self,
        src: &OperandPair,
        region: &Region,
        #[comptime] refills: Vec<Refill>,
    ) -> OperandPair {
        OperandPair {
            lhs: operand_at(&self.lhs, &src.lhs, region, refills[0]),
            rhs: operand_at(&self.rhs, &src.rhs, region, refills[1]),
            staged: comptime!(self.staged),
        }
    }

    fn inside_runs(&self) -> OperandPair {
        OperandPair {
            lhs: self.lhs.inside_run(),
            rhs: self.rhs.inside_run(),
            staged: comptime!(self.staged),
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

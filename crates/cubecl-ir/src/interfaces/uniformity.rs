use pliron::builtin::ops::ConstantOp;
use smallvec::SmallVec;

use crate::{
    PropagatesUniformity,
    interfaces::control_flow::{
        RegionBranchOpInterface, RegionBranchTerminatorOpInterface, RegionSuccessor,
    },
    prelude::*,
};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Default)]
pub enum Uniformity {
    #[default]
    Uninitialized = 4,
    Device = 3,
    Cube = 2,
    Plane = 1,
    None = 0,
}

#[op_interface]
pub trait UniformOpInterface {
    verify_op_succ!();
    fn uniformity(&self, ctx: &Context, operands: &[Uniformity]) -> Uniformity;
}

PropagatesUniformity!(ConstantOp);

#[op_interface]
pub trait UniformRegionOpInterface: RegionBranchOpInterface + SingleBlockRegionInterface {
    verify_op_succ!();
    fn result_uniformity(&self, ctx: &Context, operands: &[Uniformity]) -> Uniformity;
    fn entry_successor_region_uniformity(
        &self,
        ctx: &Context,
        operands: &[Uniformity],
    ) -> Vec<Uniformity>;
}

#[op_interface]
pub trait UniformRegionTerminatorOpInterface: RegionBranchTerminatorOpInterface {
    verify_op_succ!();
    fn all_successor_regions(&self, ctx: &Context) -> Vec<RegionSuccessor> {
        let operands = self
            .get_operation()
            .deref(ctx)
            .operands()
            .map(|_| None)
            .collect::<SmallVec<[_; 8]>>();
        self.successor_regions(ctx, &operands)
    }
    fn successor_region_uniformity(
        &self,
        ctx: &Context,
        operands: &[Uniformity],
    ) -> Vec<Uniformity>;
}

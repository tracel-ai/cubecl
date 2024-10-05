mod composite;
mod constant_prop;
mod dead_code;
mod expression_merge;
mod liveness;

pub use composite::*;
pub use dead_code::*;
pub use expression_merge::*;

use crate::AtomicCounter;

use super::Optimizer;

pub trait OptimizationPass {
    #[allow(unused)]
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
    #[allow(unused)]
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
}

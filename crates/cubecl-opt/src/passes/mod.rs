mod array_copy_propagate;
mod composite;
mod constant_prop;
mod dead_code;
mod expression_merge;
mod in_bounds_analysis;
mod index_merge;
mod integer_range_analysis;
mod liveness;

pub use array_copy_propagate::*;
pub use composite::*;
pub use constant_prop::*;
pub use dead_code::*;
pub use expression_merge::*;
pub use in_bounds_analysis::*;
pub use index_merge::*;
pub use integer_range_analysis::*;

use crate::AtomicCounter;

use super::Optimizer;

pub trait OptimizationPass {
    #[allow(unused)]
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
    #[allow(unused)]
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
}

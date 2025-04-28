mod array_copy_propagate;
mod composite;
mod constant_prop;
mod dead_code;
mod expression_merge;
mod index_merge;
mod inlined_if_to_select;
mod reduce_strength;

pub use array_copy_propagate::*;
pub use composite::*;
pub use constant_prop::*;
pub use dead_code::*;
pub use expression_merge::*;
pub use index_merge::*;
pub use inlined_if_to_select::*;
pub use reduce_strength::*;

use crate::AtomicCounter;

use super::Optimizer;

pub trait OptimizerPass {
    #[allow(unused)]
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
    #[allow(unused)]
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {}
}

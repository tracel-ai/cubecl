mod composite;
mod constant_prop;
mod dead_code;
mod disaggregate_array;
mod expression_merge;
mod index_merge;
mod inlined_if_to_select;
mod reduce_strength;

use std::any::type_name;

pub use composite::*;
pub use constant_prop::*;
pub use dead_code::*;
pub use disaggregate_array::*;
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
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }
}

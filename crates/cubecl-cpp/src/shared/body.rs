use crate::shared::Item;

use super::{Dialect, Instruction, barrier::BarrierOps};
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
#[derive(Debug, Clone)]
pub struct Body<D: Dialect> {
    pub instructions: Vec<Instruction<D>>,
    pub shared_memories: Vec<super::SharedMemory<D>>,
    pub barriers: Vec<BarrierOps<D>>,
    pub info_by_ptr: bool,
    pub has_dynamic_meta: bool,
    pub address_type: Item<D>,
}

impl<D: Dialect> Display for Body<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_bindings_body(f, self)?;

        for shared in &self.shared_memories {
            D::compile_shared_memory_declaration(f, shared)?;
        }

        for barrier in self.barriers.iter() {
            writeln!(f, "{barrier}")?;
        }

        D::compile_wmma_local_variables(f)?;

        for ops in self.instructions.iter() {
            write!(f, "{ops}")?;
        }

        Ok(())
    }
}

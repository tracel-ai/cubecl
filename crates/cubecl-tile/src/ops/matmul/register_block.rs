//! The software instruction's execution and unrolling configuration ([`RegisterBlock`]).

/// Execution and unrolling configuration for the software instruction.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct RegisterBlock {
    /// Scalar register budget for the accumulator block; blocks over budget stay rolled.
    /// Also limits the tap count eligible for gather-loop unrolling.
    pub budget: usize,
    /// Whether to generate a fast in-bounds path plus checked fallback for edge tiles.
    pub split_edge: bool,
    /// Whether to walk K as (line, component) rather than as a flat scalar walk.
    pub component_fanout: bool,
}

impl RegisterBlock {
    /// A budget with neither specialization turned on.
    pub const fn new(budget: usize) -> Self {
        Self {
            budget,
            split_edge: false,
            component_fanout: false,
        }
    }

    /// Generate the fast in-bounds path plus checked fallback for edge tiles.
    pub const fn split_edge(self) -> Self {
        Self {
            split_edge: true,
            ..self
        }
    }

    /// Walk `K` as (line, component) rather than as a flat scalar walk.
    pub const fn component_fanout(self) -> Self {
        Self {
            component_fanout: true,
            ..self
        }
    }
}

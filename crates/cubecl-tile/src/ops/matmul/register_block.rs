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
    /// Whether a block contracting an rhs lined along `K` keeps one partial a lane, summed once at
    /// the drain, rather than summing each run into a one-wide cell every step.
    pub partials: bool,
}

impl RegisterBlock {
    /// A budget with neither specialization turned on.
    pub const fn new(budget: usize) -> Self {
        Self {
            budget,
            split_edge: false,
            component_fanout: false,
            partials: false,
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

    /// Hold a run along `K` as one partial a lane: each step of the contraction is one
    /// multiply-add a cell, `vector_size` values wide, and the lanes are summed once, at the drain,
    /// rather than every step. Only for a block the drain is the one reader of: a lane holds part
    /// of a cell, so its cells are not whole values until then.
    pub const fn partials(self) -> Self {
        Self {
            partials: true,
            ..self
        }
    }
}

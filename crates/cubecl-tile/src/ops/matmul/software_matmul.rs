//! How the software matmul runs ([`SoftwareMatmul`]): its register budget and unrolling.

/// How the software matmul runs: the matrix product a unit computes in its own registers, with
/// no matrix instruction. Its register budget and unrolling choices.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SoftwareMatmul {
    /// Scalar register budget for the accumulator block; blocks over budget stay rolled.
    /// Also limits the tap count eligible for gather-loop unrolling.
    pub budget: usize,
    /// Whether to generate a fast in-bounds path plus checked fallback for edge tiles.
    pub split_edge: bool,
    /// Whether an lhs read per cell (per row in the direct nest) is walked as (vector, component)
    /// at fixed component positions rather than flat. The gather leaf walks an lhs free of the
    /// columns that way regardless.
    pub component_fanout: bool,
    /// Whether a block contracting an rhs lined along `K` keeps one partial a lane, summed once at
    /// the drain, rather than summing each run into a one-wide cell every step.
    pub partials: bool,
}

impl SoftwareMatmul {
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

    /// Walk an lhs read per cell as (vector, component) rather than flat.
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

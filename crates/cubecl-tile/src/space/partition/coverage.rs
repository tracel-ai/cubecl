//! How a level covers its tiles ([`Coverage`]): walked by one instance or distributed over a
//! scope.

use super::ComputeScope;

/// How a level covers its tiles.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Coverage {
    /// One instance steps the tiles in sequence.
    Walk,
    /// The tiles are distributed over the instances of a scope.
    Distribute(ComputeScope),
}

impl Coverage {
    /// The scope the tiles are distributed over, or `None` where one instance walks them.
    pub fn scope(self) -> Option<ComputeScope> {
        match self {
            Coverage::Walk => None,
            Coverage::Distribute(scope) => Some(scope),
        }
    }
}

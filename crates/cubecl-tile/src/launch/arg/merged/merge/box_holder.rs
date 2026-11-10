//! Who holds a box of a row reduction's output: [`BoxHolder`].

use crate::*;

/// Who holds one box of a [`ReducedRows`](crate::launch::ReducedRows) output: what its first level deals the boxes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoxHolder {
    /// A cube, its planes taking the box's rows in turn.
    Cube,
    /// A plane of a level spanning the launch, taking every row of its box.
    Plane,
}

impl BoxHolder {
    /// The holder of `level`'s boxes, each holding one box whole.
    pub(crate) fn of(level: &Level) -> Result<BoxHolder, Refusal> {
        let whole = level.shared_by().is_none()
            && level
                .axes()
                .into_iter()
                .all(|axis| level.count(axis) == Some(Count::All));
        if !whole {
            return Err(Refusal::Declined {
                why: "a row sum merges whole boxes, and this launch splits a box between holders \
                      or hands a holder several",
            });
        }
        match level.coverage() {
            Coverage::Distribute(ComputeScope::Cube) if !level.spans_launch() => {
                Ok(BoxHolder::Cube)
            }
            Coverage::Distribute(ComputeScope::Plane) if level.plane_order().is_some() => {
                Ok(BoxHolder::Plane)
            }
            _ => Err(Refusal::Declined {
                why: "a row sum's boxes are held by cubes or by planes spanning the launch",
            }),
        }
    }
}

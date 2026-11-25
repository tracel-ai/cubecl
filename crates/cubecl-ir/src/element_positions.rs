use alloc::vec::Vec;

#[cfg(all(test, any(feature = "amd", feature = "nvidia")))]
use crate::types::matrix::{MatrixIdent, MatrixShape};

/// Where each element that the lanes of a plane, or of a plane group, hold of a matrix operand lies
/// in it. A row or a column is the sum of its parts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementPositions {
    pub row: Vec<CoordinatePart>,
    pub col: Vec<CoordinatePart>,
}

impl ElementPositions {
    /// The row and the column of the `nth` element `lane` holds.
    pub fn of_nth(&self, lane: usize, nth: usize) -> (usize, usize) {
        let coordinate =
            |parts: &[CoordinatePart]| parts.iter().map(|part| part.of_nth(lane, nth)).sum();
        (coordinate(&self.row), coordinate(&self.col))
    }
}

#[cfg(all(test, any(feature = "amd", feature = "nvidia")))]
impl ElementPositions {
    /// How many times each cell of an operand, in row-major order, is held by `lanes` lanes of
    /// `elems` elements each.
    pub(crate) fn held_cells(
        &self,
        (shape, ident): (MatrixShape, MatrixIdent),
        lanes: usize,
        elems: usize,
    ) -> Vec<usize> {
        let cols = match ident {
            MatrixIdent::A => shape.k,
            MatrixIdent::B | MatrixIdent::Accumulator => shape.n,
        };
        let mut held = alloc::vec![0; shape.num_elems(ident)];
        for lane in 0..lanes {
            for nth in 0..elems {
                let (row, col) = self.of_nth(lane, nth);
                held[row * cols + col] += 1;
            }
        }
        held
    }
}

/// One part of a row or a column: how many runs of `run` lie ahead of `index`, counted again from
/// zero every `count` of them where there is one, times `stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoordinatePart {
    pub index: ElementIndex,
    pub run: usize,
    pub count: Option<usize>,
    pub stride: usize,
}

impl CoordinatePart {
    /// `index` itself.
    pub fn of(index: ElementIndex) -> Self {
        Self {
            index,
            run: 1,
            count: None,
            stride: 1,
        }
    }

    /// The same part, one step every `run` of its index.
    pub fn in_runs_of(self, run: usize) -> Self {
        Self { run, ..self }
    }

    /// The same part, starting again every `count` steps.
    pub fn counted_over(self, count: usize) -> Self {
        Self {
            count: Some(count),
            ..self
        }
    }

    /// The same part, each of its steps `stride` along.
    pub fn spaced_by(self, stride: usize) -> Self {
        Self { stride, ..self }
    }

    /// What the part adds for the `nth` element `lane` holds.
    pub fn of_nth(&self, lane: usize, nth: usize) -> usize {
        let index = match self.index {
            ElementIndex::Lane => lane,
            ElementIndex::Nth => nth,
        };
        let step = index / self.run;
        self.count.map_or(step, |count| step % count) * self.stride
    }
}

/// What a [`CoordinatePart`] counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementIndex {
    /// The unit, counted in its plane or its plane group.
    Lane,
    /// The element, counted among those its lane holds.
    Nth,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_part_counts_runs_from_zero_again_and_spaces_them() {
        let part = CoordinatePart::of(ElementIndex::Nth)
            .in_runs_of(2)
            .counted_over(3)
            .spaced_by(10);
        let places: Vec<usize> = (0..8).map(|nth| part.of_nth(5, nth)).collect();
        assert_eq!(places, [0, 0, 10, 10, 20, 20, 0, 0]);
    }

    #[test]
    fn a_coordinate_is_the_sum_of_its_parts() {
        let lane = CoordinatePart::of(ElementIndex::Lane);
        let nth = CoordinatePart::of(ElementIndex::Nth);
        let positions = ElementPositions {
            row: alloc::vec![lane.spaced_by(4), nth],
            col: alloc::vec![lane],
        };
        assert_eq!(positions.of_nth(3, 2), (14, 3));
    }
}

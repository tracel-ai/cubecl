//! [`Paging`]: which axes of an operand a page table resolves, and what an operand read
//! through one owes.

use crate::*;

/// Which axes of an operand a page table resolves: `sequence` names the page a table row lists,
/// `position` a run of `page_size` positions within one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Paging {
    pub sequence: Axis,
    pub position: Axis,
    /// Positions one page holds: the pool's extent along `position`.
    pub page_size: usize,
}

impl Paging {
    /// The buffer dims `projection` carries the two axes on, `(sequence, position)`.
    ///
    /// # Errors
    ///
    /// Either axis not one dim's identity, or both on one dim: a page is resolved digit for digit.
    pub(crate) fn dims(&self, projection: &Projection) -> Result<(usize, usize), Refusal> {
        let dim = |axis: Axis| {
            (0..projection.physical_rank())
                .find(|&pa| projection.physical_axis(pa).is_identity(axis))
                .ok_or(Refusal::PagingOfAnUnlabelledAxis(axis))
        };
        let (sequence, position) = (dim(self.sequence)?, dim(self.position)?);
        match sequence == position {
            true => Err(Refusal::PagingOfOneDim(self.sequence)),
            false => Ok((sequence, position)),
        }
    }

    /// What an operand read through a table owes, checked once for the build and the kernel both:
    /// a direct, untiled mapping over a pool of pages, moved a line at a time, its lines whole
    /// within a page and never along the pages. A packed pool is paged like a plain one: its words
    /// lie along the innermost dim, as its lines do.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first thing the table cannot resolve.
    pub(crate) fn validate(
        &self,
        projection: &Projection,
        stored_tiles: &[(Axis, usize)],
        delivery: Delivery,
        width: usize,
    ) -> Result<(), Refusal> {
        if !projection.is_direct() || projection.is_tiled() || !stored_tiles.is_empty() {
            return Err(Refusal::PagingOfAStatedLayout);
        }
        if !matches!(delivery, Delivery::SyncPerUnit | Delivery::AsyncPerUnit) {
            return Err(Refusal::PagingMovedByRuns);
        }
        let (sequence, position) = self.dims(projection)?;
        let innermost = projection.physical_rank() - 1;
        if sequence == innermost {
            return Err(Refusal::PagesAlongALine(self.sequence));
        }
        if self.page_size == 0 {
            return Err(Refusal::PagingOfAnEmptyPool);
        }
        if position == innermost && !self.page_size.is_multiple_of(width) {
            return Err(Refusal::PageSplitsALine {
                page_size: self.page_size,
                width,
            });
        }
        Ok(())
    }
}

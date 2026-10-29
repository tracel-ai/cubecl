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
    /// The paging of a pool of `shape`, its pages along the dim `sequence` labels and their
    /// positions along the one `position` labels, a page its extent there.
    ///
    /// # Errors
    ///
    /// What [`validate`](Self::validate) refuses, and a pool of no pages.
    pub(crate) fn of_pool(
        sequence: Axis,
        position: Axis,
        shape: &[usize],
        projection: &Projection,
        stored_tiles: &[(Axis, usize)],
        delivery: Delivery,
    ) -> Result<Paging, Refusal> {
        let unsized_paging = Paging {
            sequence,
            position,
            page_size: 0,
        };
        // Where the axes name no dims, `validate` refuses before it reads the page.
        let page_size = unsized_paging
            .dims(projection)
            .map_or(0, |(_, position)| shape[position]);
        let paging = Paging {
            page_size,
            ..unsized_paging
        };
        paging.validate(projection, stored_tiles, delivery)?;
        let (sequence, _) = paging.dims(projection)?;
        match shape[sequence] {
            0 => Err(Refusal::PagingOfAnEmptyPool),
            _ => Ok(paging),
        }
    }

    /// The buffer dims `projection` carries the two axes on, `(sequence, position)`.
    ///
    /// # Errors
    ///
    /// Either axis not one dim's identity, or both on one dim: a page is resolved coordinate by
    /// coordinate.
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
    /// a direct, untiled mapping over a pool of pages, moved a line at a time, its lines never
    /// along the pages. A line is whole within a page because the binding serves it whole within
    /// the dim a page spans. A packed pool is paged like a plain one: its words lie along the
    /// innermost dim, as its lines do.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first thing the table cannot resolve.
    pub(crate) fn validate(
        &self,
        projection: &Projection,
        stored_tiles: &[(Axis, usize)],
        delivery: Delivery,
    ) -> Result<(), Refusal> {
        if !projection.is_direct() || projection.is_tiled() || !stored_tiles.is_empty() {
            return Err(Refusal::PagingOfAStatedLayout);
        }
        if !matches!(delivery, Delivery::SyncPerUnit | Delivery::AsyncPerUnit) {
            return Err(Refusal::PagingMovedByRuns);
        }
        let (sequence, _) = self.dims(projection)?;
        if sequence == projection.physical_rank() - 1 {
            return Err(Refusal::PagesAlongALine(self.sequence));
        }
        match self.page_size {
            0 => Err(Refusal::PagingOfAnEmptyPool),
            _ => Ok(()),
        }
    }
}

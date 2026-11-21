//! [`Paging`]: which dimensions of an operand a page table resolves, and what an operand read
//! through one owes.

use crate::*;

/// Which dimensions of an operand a page table resolves: `sequence` names the page a table row
/// lists, `position` a run of `page_size` positions within one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Paging {
    pub sequence: Dimension,
    pub position: Dimension,
    /// Positions one page holds: the pool's extent along `position`.
    pub page_size: usize,
}

impl Paging {
    /// The paging of a pool of `shape`, its pages along the axis `sequence` labels and their
    /// positions along the one `position` labels, a page its extent there.
    ///
    /// # Errors
    ///
    /// What [`validate`](Self::validate) refuses, and a pool of no pages.
    pub(crate) fn of_pool(
        sequence: Dimension,
        position: Dimension,
        shape: &[usize],
        projection: &Projection,
        stored_tiles: &[(Dimension, usize)],
        delivery: Delivery,
    ) -> Result<Paging, Decline> {
        let unsized_paging = Paging {
            sequence,
            position,
            page_size: 0,
        };
        // Where the dimensions name no axes, `validate` refuses before it reads the page.
        let page_size = unsized_paging
            .axes(projection)
            .map_or(0, |(_, position)| shape[position]);
        let paging = Paging {
            page_size,
            ..unsized_paging
        };
        paging.validate(projection, stored_tiles, delivery)?;
        let (sequence, _) = paging.axes(projection)?;
        match shape[sequence] {
            0 => Err(Decline::PagingOfAnEmptyPool),
            _ => Ok(paging),
        }
    }

    /// The buffer axes `projection` carries the two dimensions on, `(sequence, position)`.
    ///
    /// # Errors
    ///
    /// Either dimension not one axis's identity, or both on one axis: a page is resolved coordinate
    /// by coordinate.
    pub(crate) fn axes(&self, projection: &Projection) -> Result<(usize, usize), Decline> {
        let axis = |dimension: Dimension| {
            (0..projection.physical_rank())
                .find(|&pa| projection.physical_axis(pa).is_identity(dimension))
                .ok_or(Decline::PagingOfAnUnlabelledDimension(dimension))
        };
        let (sequence, position) = (axis(self.sequence)?, axis(self.position)?);
        match sequence == position {
            true => Err(Decline::PagingOfOneAxis(self.sequence)),
            false => Ok((sequence, position)),
        }
    }

    /// What an operand read through a table owes, checked once for the build and the kernel both:
    /// a direct, untiled mapping over a pool of pages, moved a line at a time, its lines never
    /// along the pages. A line is whole within a page because the binding serves it whole within
    /// the axis a page spans. A packed pool is paged like a plain one: its words lie along the
    /// innermost axis, as its lines do.
    ///
    /// # Errors
    ///
    /// The [`Decline`] naming the first thing the table cannot resolve.
    pub(crate) fn validate(
        &self,
        projection: &Projection,
        stored_tiles: &[(Dimension, usize)],
        delivery: Delivery,
    ) -> Result<(), Decline> {
        if !projection.is_direct() || projection.is_tiled() || !stored_tiles.is_empty() {
            return Err(Decline::PagingOfAStatedLayout);
        }
        if delivery.issuer() != Issuer::Unit {
            return Err(Decline::PagingMovedByRuns);
        }
        let (sequence, _) = self.axes(projection)?;
        if sequence == projection.physical_rank() - 1 {
            return Err(Decline::PagesAlongALine(self.sequence));
        }
        match self.page_size {
            0 => Err(Decline::PagingOfAnEmptyPool),
            _ => Ok(()),
        }
    }
}

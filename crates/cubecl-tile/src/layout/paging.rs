//! A pool of fixed-size pages read as sequences: [`Paging`] states it, [`PageTable`] resolves it.
//!
//! The buffer is a pool whose `sequence` dim holds pages and whose `position` dim a page's
//! positions. An operand bound over it is read as the sequences a table names: sequence `b`'s
//! position `s` lies in page `table[b, s / page_size]`, at row `s % page_size`. The projection stays
//! direct over the logical `[…, sequence, …, position, …]` axes, so every window, walk and bound
//! is the logical one; the table is resolved only where a coordinate becomes an offset
//! ([`BufferLayout`](super::projection::BufferLayout)), at every offset taken, a masked cell's
//! included. The pool is bound as it lies; the operand's extents along the two axes are the
//! table's, read off it at launch, so nothing states them twice. A coordinate past the table reads
//! the pool's first page, and a table entry past the pool its last: a table is device data a
//! launch may never have written.
//!
//! The table is read as each window is cut, not as each line is: a window pins the page its origin
//! lies in, and one whose extent is a single sequence and at most a page of positions the next page
//! too, its lines offset from the two pages' starts without touching the table. A window proved
//! inside one page pins it alone, and a window cut inside that one keeps the pin.

use cubecl::prelude::*;

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
    /// a direct, untiled mapping over a pool of pages, never packed, moved a line at a time, its
    /// lines whole within a page and never along the pages.
    ///
    /// # Errors
    ///
    /// The [`Refusal`] naming the first thing the table cannot resolve.
    pub(crate) fn validate(
        &self,
        projection: &Projection,
        stored_tiles: &[(Axis, usize)],
        packing: Packing,
        delivery: Delivery,
        width: usize,
    ) -> Result<(), Refusal> {
        if !projection.is_direct() || projection.is_tiled() || !stored_tiles.is_empty() {
            return Err(Refusal::PagingOfAStatedLayout);
        }
        if packing != Packing::Plain
            || !matches!(delivery, Delivery::SyncPerUnit | Delivery::AsyncPerUnit)
        {
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

/// What a window's extent proves of the pages its lines lie in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Spans {
    /// Nothing: a line leaving the pinned page reads its own off the table.
    Any,
    /// One sequence and at most a page of positions: the pinned page, or the next.
    TwoPages,
    /// One sequence inside the pinned page.
    OnePage,
}

impl Spans {
    /// What this proves of the same extent moved by a runtime amount, which loses the origin's
    /// alignment.
    fn shifted(self) -> Spans {
        match self {
            Spans::OnePage => Spans::TwoPages,
            spans => spans,
        }
    }
}

/// A window's extent along the paged axes, `None` where no level fixes it, and what its origin is
/// aligned to along the positions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PagedWindow {
    sequences: Option<usize>,
    positions: Option<usize>,
    origin_alignment: usize,
}

impl PagedWindow {
    /// What this window proves over pages of `page_size` positions: one page where its positions
    /// fit in the alignment its origin and the pages share.
    fn spans(&self, page_size: usize) -> Spans {
        match (self.sequences, self.positions) {
            (Some(1), Some(positions)) if positions <= gcd(self.origin_alignment, page_size) => {
                Spans::OnePage
            }
            (Some(1), Some(positions)) if positions <= page_size => Spans::TwoPages,
            _ => Spans::Any,
        }
    }
}

/// What a window cut from the one a table is pinned for does with that pin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Pin {
    /// The same pin: the window lies in the page this table pinned.
    Kept,
    /// Its own pin, for a window spanning pages as the variant says.
    Resolved(Spans),
}

/// How a window was cut from the one before, which decides what it keeps of that one's pages.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum WindowCut {
    /// The same origin and extent: handed to other units, or bounded short of its end.
    Unmoved,
    /// A level's tile, its extent along each dim in loads where the level fixes it; `nested` where
    /// the window is paged and every tile lies inside the window it was cut from.
    Level {
        edges: Vec<Option<usize>>,
        nested: bool,
    },
    /// Moved by a runtime amount: a `within` from a runtime start, or a gathered descent.
    Shifted,
}

/// A page table as a kernel reads it: `u32`, `[sequences, pages_per_sequence]`, read by stride.
/// Its shape is the operand's: the sequences it lists and their positions, `page_size` a column.
/// It holds at least one entry, which [`lookup`](Self::lookup) reads for every window.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct PageTable {
    entries: Box<[u32]>,
    sequence_stride: u32,
    page_stride: u32,
    /// The table's rows: a sequence past them is outside the operand, and reads the pool's first
    /// page.
    pub(crate) sequences: u32,
    /// The table's columns: a position past them is likewise outside, and reads the first page.
    pub(crate) pages_per_sequence: u32,
    /// The pool's last page: a table entry past it reads it, since a table is device data a
    /// launch may never have written.
    last_page: u32,
    /// The page a window's origin lies in, resolved once as the window is cut
    /// ([`pinned_at`](Self::pinned_at)): a line in the same sequence and the same column of the
    /// table reads it without loading from the table.
    pin_sequence: u32,
    pin_column: u32,
    pin_page: u32,
    /// The page of the column after the pin's, where the window [`Spans::TwoPages`].
    next_page: u32,
    /// Where the pin's column starts, in positions.
    pin_start: u32,
    /// What a line of the pinned page, or the next, adds to its offset as one sequence's.
    pin_base: u32,
    next_base: u32,
    /// What the window's extent proves of the pages its lines lie in.
    #[cube(comptime)]
    spans: Spans,
    /// What the window's origin along the positions is aligned to, in positions: 0 for the origin
    /// of the whole operand (aligned to everything), 1 where nothing is known.
    #[cube(comptime)]
    origin_alignment: usize,
    #[cube(comptime)]
    pub(crate) page_size: usize,
    /// The buffer dim the pages lie along.
    #[cube(comptime)]
    pub(crate) sequence_dim: usize,
    /// The buffer dim a page's positions lie along.
    #[cube(comptime)]
    pub(crate) position_dim: usize,
    /// Positions one step of the position dim's coordinate covers: a line's width where the
    /// positions run along the innermost dim, which is addressed in lines; one elsewhere.
    #[cube(comptime)]
    unit: usize,
}

#[cube]
impl PageTable {
    /// `table` resolving `paging`'s axes, which `projection` carries, over a pool of `pool_pages`
    /// pages, an operand read `width` values a line along its innermost dim.
    pub(crate) fn new(
        table: &Tensor<u32>,
        pool_pages: u32,
        #[comptime] paging: Paging,
        #[comptime] projection: Projection,
        #[comptime] width: usize,
    ) -> PageTable {
        let (sequence_dim, position_dim) = comptime!(
            paging
                .dims(&projection)
                .unwrap_or_else(|refusal| panic!("PageTable: {refusal}"))
        );
        let unit = comptime!(match position_dim == projection.physical_rank() - 1 {
            true => width,
            false => 1,
        });
        // Only `[sequences, pages_per_sequence]` names a page per sequence and column; any other
        // rank reads as a table of no rows, so every line is masked rather than a dim past the
        // table's rank read for its extent.
        let mut sequence_stride = 0u32;
        let mut page_stride = 0u32;
        let mut sequences = 0u32;
        let mut pages_per_sequence = 0u32;
        if table.rank() == 2 {
            sequence_stride = table.stride(0) as u32;
            page_stride = table.stride(1) as u32;
            sequences = table.shape(0) as u32;
            pages_per_sequence = table.shape(1) as u32;
        }
        let unpinned = PageTable {
            entries: unsafe { table.as_slice().as_boxed_unchecked() },
            sequence_stride,
            page_stride,
            sequences,
            pages_per_sequence,
            // A bound pool holds a page at least; one stated by hand is kept off underflow.
            last_page: pool_pages.max_with(1u32).minus(1u32),
            pin_sequence: 0u32,
            pin_column: 0u32,
            pin_page: 0u32,
            next_page: 0u32,
            pin_start: 0u32,
            pin_base: 0u32,
            next_base: 0u32,
            spans: comptime!(Spans::Any),
            origin_alignment: comptime!(0usize),
            page_size: comptime!(paging.page_size),
            sequence_dim,
            position_dim,
            unit,
        };
        unpinned.pinned(0u32, 0u32, comptime!(Spans::Any), comptime!(0usize))
    }

    /// This table for the window whose origin is `origin`, cut from the one this table was pinned
    /// for as `cut` says. A window inside the one page of the window it was cut from keeps that
    /// page; any other resolves its own.
    pub(crate) fn pinned_at(
        &self,
        origin: &Coords<i32>,
        #[comptime] cut: WindowCut,
        sequence_stride: u32,
        position_stride: u32,
    ) -> PageTable {
        let origin_alignment = comptime!(match &cut {
            WindowCut::Unmoved => self.origin_alignment,
            WindowCut::Shifted => 1,
            WindowCut::Level { edges, .. } => match edges[self.position_dim] {
                Some(loads) => gcd(self.origin_alignment, loads * self.unit),
                None => self.origin_alignment,
            },
        });
        let pin = comptime!(match cut {
            WindowCut::Unmoved => Pin::Kept,
            WindowCut::Shifted => Pin::Resolved(self.spans.shifted()),
            WindowCut::Level { nested: true, .. } if self.spans == Spans::OnePage => Pin::Kept,
            WindowCut::Level { edges, .. } => Pin::Resolved(
                PagedWindow {
                    sequences: edges[self.sequence_dim],
                    positions: edges[self.position_dim].map(|loads| loads * self.unit),
                    origin_alignment,
                }
                .spans(self.page_size),
            ),
        });
        match comptime!(pin) {
            Pin::Kept => self.aligned(origin_alignment),
            Pin::Resolved(spans) => self.repinned(
                origin,
                spans,
                origin_alignment,
                sequence_stride,
                position_stride,
            ),
        }
    }

    /// This table pinned at the page `origin` lies in, for a window spanning pages as `spans` says.
    fn repinned(
        &self,
        origin: &Coords<i32>,
        #[comptime] spans: Spans,
        #[comptime] origin_alignment: usize,
        sequence_stride: u32,
        position_stride: u32,
    ) -> PageTable {
        let page_size = comptime!(self.page_size as u32);
        let unit = comptime!(self.unit as u32);
        let sequence = origin.at(comptime!(self.sequence_dim)).cast::<u32>();
        let position = origin
            .at(comptime!(self.position_dim))
            .cast::<u32>()
            .times(unit);
        let mut table = self.pinned(
            sequence,
            position.divided_by(page_size),
            spans,
            origin_alignment,
        );
        // Wrapping: a base is only ever added to the offset of a line past its column's start.
        let start = table.pin_start.divided_by(unit);
        let page_digits = comptime!((self.page_size / self.unit) as u32);
        table.pin_base = table
            .pin_page
            .times(sequence_stride)
            .minus(start.times(position_stride));
        table.next_base = table
            .next_page
            .times(sequence_stride)
            .minus(start.plus(page_digits).times(position_stride));
        table
    }

    /// This table pinned at the page column `column` of sequence `sequence` lists.
    fn pinned(
        &self,
        sequence: u32,
        column: u32,
        #[comptime] spans: Spans,
        #[comptime] origin_alignment: usize,
    ) -> PageTable {
        let next_page = match comptime!(spans) {
            Spans::TwoPages => self.lookup(sequence, column.plus(1u32)),
            Spans::Any | Spans::OnePage => 0u32,
        };
        PageTable {
            entries: self.entries.clone(),
            sequence_stride: self.sequence_stride,
            page_stride: self.page_stride,
            sequences: self.sequences,
            pages_per_sequence: self.pages_per_sequence,
            last_page: self.last_page,
            pin_sequence: sequence,
            pin_column: column,
            pin_page: self.lookup(sequence, column),
            next_page,
            pin_start: column.times(comptime!(self.page_size as u32)),
            pin_base: 0u32,
            next_base: 0u32,
            spans,
            origin_alignment,
            page_size: comptime!(self.page_size),
            sequence_dim: comptime!(self.sequence_dim),
            position_dim: comptime!(self.position_dim),
            unit: comptime!(self.unit),
        }
    }

    /// This table, its pin unchanged, for a window whose origin is aligned to `origin_alignment`.
    fn aligned(&self, #[comptime] origin_alignment: usize) -> PageTable {
        PageTable {
            entries: self.entries.clone(),
            sequence_stride: self.sequence_stride,
            page_stride: self.page_stride,
            sequences: self.sequences,
            pages_per_sequence: self.pages_per_sequence,
            last_page: self.last_page,
            pin_sequence: self.pin_sequence,
            pin_column: self.pin_column,
            pin_page: self.pin_page,
            next_page: self.next_page,
            pin_start: self.pin_start,
            pin_base: self.pin_base,
            next_base: self.next_base,
            spans: comptime!(self.spans),
            origin_alignment,
            page_size: comptime!(self.page_size),
            sequence_dim: comptime!(self.sequence_dim),
            position_dim: comptime!(self.position_dim),
            unit: comptime!(self.unit),
        }
    }

    /// Where line `digits` lies, as digits and a base its offset adds past them: a window of at
    /// most two pages keeps one sequence's digits, so its lines' offsets stay affine, and adds its
    /// page's base; any other is resolved line by line.
    pub(crate) fn located(&self, digits: &Coords<u32>) -> (Coords<u32>, u32) {
        match comptime!(self.spans) {
            Spans::OnePage => (self.pinned_digits(digits), self.pin_base),
            Spans::TwoPages => {
                let page_size = comptime!(self.page_size as u32);
                let unit = comptime!(self.unit as u32);
                let pinned_digits = self.pinned_digits(digits);
                let position = pinned_digits.at(comptime!(self.position_dim)).times(unit);
                let pinned = position.minus(self.pin_start) < page_size;
                (pinned_digits, select(pinned, self.pin_base, self.next_base))
            }
            Spans::Any => (self.resolve(digits), 0u32),
        }
    }

    /// `digits` with the sequence's digit zeroed, since one sequence's lines are offset from their
    /// pages' bases, and the position held at or past the pin's start: a checked read addresses a
    /// line past the window's bound at the bound's edge, which may lie before the pinned page, and
    /// drops what it reads there.
    fn pinned_digits(&self, digits: &Coords<u32>) -> Coords<u32> {
        let first = self.pin_start.divided_by(comptime!(self.unit as u32));
        let mut pinned = Coords::<u32>::new();
        #[unroll]
        for pa in 0..digits.len() {
            if comptime!(pa == self.sequence_dim) {
                pinned.push(0u32);
            } else if comptime!(pa == self.position_dim) {
                pinned.push(digits.at(pa).max_with(first));
            } else {
                pinned.push(digits.at(pa));
            }
        }
        pinned
    }

    /// The page column `column` of sequence `sequence` lists, clamped to the pool.
    ///
    /// A coordinate outside the table, which only a masked read asks for before the mask drops
    /// it, reads the pool's first page and never loads past the table.
    fn lookup(&self, sequence: u32, column: u32) -> u32 {
        // No branch, so a window's reads schedule around the load: the coordinate is clamped
        // into the table, which holds an entry, and the entry dropped where it was outside.
        let inside = sequence < self.sequences && column < self.pages_per_sequence;
        let at = sequence
            .min_with(self.sequences.max_with(1u32).minus(1u32))
            .times(self.sequence_stride)
            .plus(
                column
                    .min_with(self.pages_per_sequence.max_with(1u32).minus(1u32))
                    .times(self.page_stride),
            );
        let entry = self.entries[at.cast::<usize>()].min_with(self.last_page);
        select(inside, entry, 0u32)
    }

    /// `pool`, the geometry of the pool as it lies, as the sequences this table lists: the table's
    /// rows along the sequence dim and their positions along the position dim, over the pool's
    /// strides.
    pub(crate) fn sequences_over(&self, pool: &RuntimeGeometry) -> RuntimeGeometry {
        let page_size = comptime!(self.page_size as u32);
        let mut sequences = RuntimeGeometry::new();
        #[unroll]
        for pa in 0..pool.shape.len() {
            let extent = if comptime!(pa == self.sequence_dim) {
                self.sequences
            } else if comptime!(pa == self.position_dim) {
                self.pages_per_sequence.times(page_size)
            } else {
                pool.shape.at(pa)
            };
            sequences.push(extent, pool.strides.at(pa));
        }
        sequences
    }

    /// `digits`, a logical coordinate along each buffer dim, with the sequence's digit replaced
    /// by the page its position lies in and the position's by its row there: the pinned page
    /// where the line lies in it, else the table's [`lookup`](Self::lookup).
    fn resolve(&self, digits: &Coords<u32>) -> Coords<u32> {
        let page_size = comptime!(self.page_size as u32);
        let unit = comptime!(self.unit as u32);
        let sequence = digits.at(comptime!(self.sequence_dim));
        let position = digits.at(comptime!(self.position_dim)).times(unit);
        let column = position.divided_by(page_size);
        let mut page = self.pin_page;
        if sequence != self.pin_sequence || column != self.pin_column {
            page = self.lookup(sequence, column);
        }
        let mut resolved = Coords::<u32>::new();
        #[unroll]
        for pa in 0..digits.len() {
            if comptime!(pa == self.sequence_dim) {
                resolved.push(page);
            } else if comptime!(pa == self.position_dim) {
                resolved.push(position.remainder(page_size).divided_by(unit));
            } else {
                resolved.push(digits.at(pa));
            }
        }
        resolved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(sequences: usize, positions: usize, origin_alignment: usize) -> PagedWindow {
        PagedWindow {
            sequences: Some(sequences),
            positions: Some(positions),
            origin_alignment,
        }
    }

    /// One page only where the alignment the origin and the pages share holds the whole extent:
    /// a step of 16 from 0 or a multiple of 16 stays in a page of 64; from a multiple of 4, or as
    /// 48 positions, it may cross into the next.
    #[test]
    fn a_window_is_inside_one_page_only_where_its_alignment_proves_it() {
        assert_eq!(window(1, 16, 0).spans(64), Spans::OnePage);
        assert_eq!(window(1, 16, 16).spans(64), Spans::OnePage);
        assert_eq!(window(1, 64, 128).spans(64), Spans::OnePage);
        assert_eq!(window(1, 16, 4).spans(64), Spans::TwoPages);
        assert_eq!(window(1, 48, 48).spans(64), Spans::TwoPages);
        assert_eq!(window(1, 128, 0).spans(64), Spans::Any);
        assert_eq!(window(2, 4, 0).spans(64), Spans::Any);
        let unfixed = PagedWindow {
            sequences: Some(1),
            positions: None,
            origin_alignment: 0,
        };
        assert_eq!(unfixed.spans(64), Spans::Any);
    }

    /// A runtime shift keeps the extent, so at most two pages, but loses the alignment.
    #[test]
    fn a_shifted_window_keeps_its_extent_not_its_page() {
        assert_eq!(Spans::OnePage.shifted(), Spans::TwoPages);
        assert_eq!(Spans::TwoPages.shifted(), Spans::TwoPages);
        assert_eq!(Spans::Any.shifted(), Spans::Any);
    }
}

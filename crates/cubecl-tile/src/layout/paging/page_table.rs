//! [`PageTable`]: a page table as a kernel reads it, pinned at each window cut.

use cubecl::prelude::*;

use super::window::{PagedWindow, Pin, Spans};
use crate::*;

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
    /// The entry this window was cut with, resolved once at the cut
    /// ([`pinned_at`](Self::pinned_at)): a line in the same sequence and column reads its page from
    /// here without loading the table. Replaced at each cut that resolves a new page.
    pin: PageEntry,
    /// Where the pin's column starts, in positions.
    pin_start: u32,
    /// What a line of the pinned page, or the next where the window [`Spans::TwoPages`], adds to
    /// its offset as one sequence's.
    pin_base: u32,
    next_base: u32,
    /// An entry handed in by [`Tile::with_page_entry`](crate::Tile::with_page_entry), consulted by
    /// every cut that resolves its page and kept across cuts, unlike [`pin`](Self::pin). It holds
    /// one page, so the page after it, which a [`Spans::TwoPages`] window also needs, is still
    /// loaded. Absent at compile time when none is handed, so the cut emits no compare.
    handed: ComptimeOption<PageEntry>,
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
            pin: PageEntry::unlisted(),
            pin_start: 0u32,
            pin_base: 0u32,
            next_base: 0u32,
            handed: ComptimeOption::new_None(),
            spans: comptime!(Spans::Any),
            origin_alignment: comptime!(0usize),
            page_size: comptime!(paging.page_size),
            sequence_dim,
            position_dim,
            unit,
        };
        let (first, no_stride) = (0u32, 0u32);
        unpinned.pinned(
            first,
            first,
            comptime!(Spans::Any),
            comptime!(0usize),
            no_stride,
            no_stride,
        )
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
            Pin::Kept => self.with_alignment_and_entry(origin_alignment, self.handed.clone()),
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
        let sequence = origin.at(comptime!(self.sequence_dim)).cast::<u32>();
        let position = origin
            .at(comptime!(self.position_dim))
            .cast::<u32>()
            .times(comptime!(self.unit as u32));
        self.pinned(
            sequence,
            position.divided_by(comptime!(self.page_size as u32)),
            spans,
            origin_alignment,
            sequence_stride,
            position_stride,
        )
    }

    /// This table pinned at the page column `column` of sequence `sequence` lists, with the bases
    /// its lines add to their offsets.
    fn pinned(
        &self,
        sequence: u32,
        column: u32,
        #[comptime] spans: Spans,
        #[comptime] origin_alignment: usize,
        sequence_stride: u32,
        position_stride: u32,
    ) -> PageTable {
        let unit = comptime!(self.unit as u32);
        let page = self.cut_page(sequence, column);
        let next_page = match comptime!(spans) {
            Spans::TwoPages => self.lookup(sequence, column.plus(1u32)),
            Spans::Any | Spans::OnePage => 0u32,
        };
        let pin_start = column.times(comptime!(self.page_size as u32));
        // Wrapping: a base is only ever added to the offset of a line past its column's start.
        let start = pin_start.divided_by(unit);
        let page_digits = comptime!((self.page_size / self.unit) as u32);
        PageTable {
            entries: self.entries.clone(),
            sequence_stride: self.sequence_stride,
            page_stride: self.page_stride,
            sequences: self.sequences,
            pages_per_sequence: self.pages_per_sequence,
            last_page: self.last_page,
            pin: PageEntry {
                sequence,
                column,
                page,
            },
            pin_start,
            pin_base: page
                .times(sequence_stride)
                .minus(start.times(position_stride)),
            next_base: next_page
                .times(sequence_stride)
                .minus(start.plus(page_digits).times(position_stride)),
            handed: self.handed.clone(),
            spans,
            origin_alignment,
            page_size: comptime!(self.page_size),
            sequence_dim: comptime!(self.sequence_dim),
            position_dim: comptime!(self.position_dim),
            unit: comptime!(self.unit),
        }
    }

    /// The page `column` of sequence `sequence` lists: the handed entry where it names that cell,
    /// else the table's [`lookup`](Self::lookup).
    fn cut_page(&self, sequence: u32, column: u32) -> u32 {
        #[comptime]
        match &self.handed {
            ComptimeOption::Some(handed) => self.page_from(handed, sequence, column),
            ComptimeOption::None => self.lookup(sequence, column),
        }
    }

    /// The page `column` of sequence `sequence` lists: `entry`'s where it names that cell, else
    /// the table's.
    fn page_from(&self, entry: &PageEntry, sequence: u32, column: u32) -> u32 {
        let mut page = entry.page;
        if sequence != entry.sequence || column != entry.column {
            page = self.lookup(sequence, column);
        }
        page
    }

    /// The entry this table's window was cut with.
    pub(crate) fn pinned_entry(&self) -> PageEntry {
        self.pin.clone()
    }

    /// This table taking its pages from `entry` where a cut names its cell.
    pub(crate) fn with_page_entry(&self, entry: &PageEntry) -> PageTable {
        self.with_alignment_and_entry(
            comptime!(self.origin_alignment),
            ComptimeOption::new_Some(entry.clone()),
        )
    }

    /// This table, its window unchanged, with the origin alignment and the handed entry given.
    fn with_alignment_and_entry(
        &self,
        #[comptime] origin_alignment: usize,
        handed: ComptimeOption<PageEntry>,
    ) -> PageTable {
        PageTable {
            entries: self.entries.clone(),
            sequence_stride: self.sequence_stride,
            page_stride: self.page_stride,
            sequences: self.sequences,
            pages_per_sequence: self.pages_per_sequence,
            last_page: self.last_page,
            pin: self.pin.clone(),
            pin_start: self.pin_start,
            pin_base: self.pin_base,
            next_base: self.next_base,
            handed,
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
        let page = self.page_from(&self.pin, sequence, column);
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

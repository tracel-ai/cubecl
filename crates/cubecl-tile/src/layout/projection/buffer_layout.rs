//! The runtime half of a [`Projection`] over one buffer: its shape and strides, dotted in-kernel.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// In-kernel twin of cubecl's `TiledViewLayout` over a buffer's physical coordinates, one per
/// buffer axis: their offset once paged, placed and dotted with the strides, with no division.
/// `projection` is [`Projection::positional`]'s synthetic map, which the inverse reads and which
/// says how a coordinate splits into the pieces a storage-tiled buffer keeps it as
/// ([`pieces`](BufferLayout::pieces)).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct BufferLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// Where a stage keeps each line of a block row.
    #[cube(comptime)]
    pub(crate) rows: RowArrangement,
    /// The table a pool of pages is read as sequences through ([`Paging`]); `None` for a buffer
    /// read where it lies.
    pub(crate) paging: ComptimeOption<PageTable>,
}

#[cube]
impl Layout for BufferLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime!(self.projection.physical_rank());
        let mut physical = Coords::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            physical.push(pos[pa]);
        }
        let (physical, base) = self.paged(physical);
        let physical = self.placed(&physical);
        let mut terms = Sequence::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            terms.push(physical.at(pa).times(self.physical_strides.at(pa)));
        }
        terms
            .sum(comptime!((0..rank).collect::<Vec<_>>()))
            .plus(base)
            .cast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.physical_shape.to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.physical_shape.within(pos)
    }
}

#[cube]
impl BufferLayout {
    /// The split of a coordinate into the pieces a storage-tiled buffer keeps it as: each piece
    /// the digit of the coordinate plus its entry of `starts` that the piece's axis stores, over
    /// a coordinate box `shape`. What a reader holding coordinates composes this layout under.
    pub(crate) fn pieces(&self, starts: &Coords<u32>, shape: Coords<u32>) -> ProjectionInKernel {
        let positional = comptime!(self.projection.clone());
        let rank = comptime!(positional.coordinate_rank());
        let radices = self.radices();
        let split = comptime!(positional.untiled().pieces_as_digits(&positional, &radices));
        let space = comptime!(
            Space::new(
                &(0..rank)
                    .map(|c| (Dimension(c as u8), 1usize))
                    .collect::<Vec<_>>()
            )
            .all_dynamic()
        );
        let digits = comptime!(split.digit_dimensions());
        let mut digit_starts = Coords::<u32>::new();
        #[unroll]
        for d in 0..digits.len() {
            digit_starts.push(starts.at(comptime!(digits[d].0 as usize)));
        }
        let map = RuntimeMap {
            coefficients: Coords::<u32>::new(),
            residues: Coords::constant(comptime!(vec![0; positional.physical_rank()])),
            digit_starts,
            digit_radices: self.digit_radices(&split),
        };
        ProjectionInKernel::new(shape, map, space, split, 1usize)
    }

    /// [`pieces`](BufferLayout::pieces) over the buffer's whole box from its origin: the split
    /// a reader holding the buffer's own coordinates composes this layout under.
    pub(crate) fn split(&self) -> ProjectionInKernel {
        let rank = comptime!(self.projection.coordinate_rank());
        self.pieces(
            &Coords::constant(comptime!(vec![0; rank])),
            logical_extent(&self.projection, &self.physical_shape),
        )
    }

    /// The line the buffer's own coordinate `coords` lies at: split into its pieces first where
    /// the buffer stores it as several.
    pub(crate) fn coordinate_offset(&self, coords: CoordsDyn) -> usize {
        if comptime!(self.projection.is_tiled()) {
            self.to_source_pos(self.split().to_source_pos(coords))
        } else {
            self.to_source_pos(coords)
        }
    }

    /// Each buffer axis's place value and extent in loads: [`Radix::Static`] where the extents
    /// are constants, as a stated tile's are, [`Radix::Dynamic`] where they are read off the
    /// buffer.
    pub(crate) fn radices(&self) -> comptime_type!(Vec<(Radix, Radix)>) {
        let positional = comptime!(self.projection.clone());
        let mut radices = comptime!(Vec::new());
        #[unroll]
        for pa in 0..comptime!(positional.physical_rank()) {
            let finer = comptime!(self.projection.finer_pieces(pa));
            let place = self.physical_shape.product(finer).constant();
            let extent = self.physical_shape.at(pa).constant();
            comptime!(radices.push((
                place.map_or(Radix::Dynamic, |p| Radix::Static(p as usize)),
                extent.map_or(Radix::Dynamic, |e| Radix::Static(e as usize)),
            )));
        }
        radices
    }

    /// The [`Radix::Dynamic`] radices `digits`, this buffer's pieces read as digits
    /// ([`Projection::pieces_as_digits`] of [`radices`](BufferLayout::radices)), read off the
    /// buffer, in [`Projection::digit_radix_index`] order.
    pub(crate) fn digit_radices(&self, #[comptime] digits: &Projection) -> Coords<u32> {
        let mut dynamic = Coords::<u32>::new();
        #[unroll]
        for pa in 0..comptime!(digits.physical_rank()) {
            let (place_at, extent_at) = comptime!(digits.digit_radix_index(pa));
            if comptime!(place_at.is_some()) {
                dynamic.push(
                    self.physical_shape
                        .product(comptime!(self.projection.finer_pieces(pa))),
                );
            }
            if comptime!(extent_at.is_some()) {
                dynamic.push(self.physical_shape.at(pa));
            }
        }
        dynamic
    }

    /// The logical coordinate line `i` of this buffer holds.
    pub(crate) fn line_coords(&self, i: usize) -> CoordsDyn {
        let physical = self.placed(&self.line_physical(i));
        self.logical_of_physical(&physical)
    }

    /// Where line `i` of this buffer sits: `i` itself unless rows are padded.
    pub(crate) fn line_offset(&self, i: usize) -> usize {
        if comptime!(self.rows.is_pitched()) {
            let physical = self.line_physical(i);
            let mut offset = 0u32;
            #[unroll]
            for pa in 0..physical.len() {
                offset = offset.plus(physical.at(pa).times(self.physical_strides.at(pa)));
            }
            offset.cast::<usize>()
        } else {
            i
        }
    }

    /// The logical coordinate of line `i` counted row-major over the logical extents: the
    /// source's order, whatever blocking or swizzle places the line here.
    pub(crate) fn row_major_coords(&self, i: usize) -> CoordsDyn {
        let rank = comptime!(self.projection.logical_rank());
        let extent = logical_extent(&self.projection, &self.physical_shape);
        let x = i.cast::<u32>();
        let mut coords = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            let inner = comptime!((p + 1..rank).collect::<Vec<_>>());
            let quot = x.divided_by(extent.product(inner));
            let coord = if p == 0 {
                quot
            } else {
                quot.remainder(extent.at(p))
            };
            coords.push(coord);
        }
        coords
    }

    /// The physical coordinates line `i` sits at, one per physical axis, row-major over its
    /// extents.
    fn line_physical(&self, i: usize) -> Coords<u32> {
        let x = i.cast::<u32>();
        let rank = self.physical_shape.len();
        let mut physical = Coords::<u32>::new();
        #[unroll]
        for pa in 0..rank {
            physical.push(
                x.divided_by(
                    self.physical_shape
                        .product(comptime!(((pa + 1)..rank).collect::<Vec<_>>())),
                )
                .remainder(self.physical_shape.at(pa)),
            );
        }
        physical
    }

    /// The logical coordinate producing `physical` under this layout's projection, inverting
    /// [`to_source_pos`](Layout::to_source_pos). Requires [`Projection::is_invertible`].
    fn logical_of_physical(&self, physical: &Coords<u32>) -> CoordsDyn {
        comptime!(assert!(
            self.projection.is_invertible(),
            "Projection::logical_of_physical: not invertible (an affine term mixes several logical \
             coordinates into one physical cell)"
        ));
        let mut out = CoordsDyn::new();
        #[unroll]
        for i in 0..comptime!(self.projection.logical_rank()) {
            let dimension = comptime!(self.projection.dimensions()[i]);
            let carriers = comptime!(self.projection.carriers(dimension));
            let picks = comptime!((0..carriers.len()).collect::<Vec<_>>());
            let mut parts = Sequence::<u32>::new();
            #[unroll]
            for k in 0..carriers.len() {
                let pa = comptime!(carriers[k]);
                // Each coordinate weighs the block it sits above, the same finer extents that
                // stripped it.
                let (finer, _) = comptime!(self.projection.carried_by(pa, dimension));
                parts.push(
                    physical
                        .at(pa)
                        .times(self.physical_shape.product(finer.to_vec())),
                );
            }
            out.push(parts.sum(picks));
        }
        out
    }

    /// This layout for a window whose origin is `origin`, cut as `cut` says
    /// ([`PageTable::pinned_at`]): the pages the window reads resolved once where the buffer is a
    /// pool of pages, unchanged elsewhere.
    pub(crate) fn pinned_at(
        &self,
        origin: &Coords<i32>,
        #[comptime] cut: WindowCut,
    ) -> BufferLayout {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => self.with_paging(ComptimeOption::new_Some(
                table.pinned_at(
                    origin,
                    cut,
                    self.physical_strides
                        .at(comptime!(table.list.sequence_axis)),
                    self.physical_strides
                        .at(comptime!(table.list.position_axis)),
                ),
            )),
            ComptimeOption::None => self.clone(),
        }
    }

    /// The entry this layout's window was cut with, [`PageEntry::unlisted`] where the buffer is not
    /// a pool of pages.
    pub(crate) fn pinned_entry(&self) -> PageEntry {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => table.pinned_entry(),
            ComptimeOption::None => PageEntry::unlisted(),
        }
    }

    /// This layout taking its pages from `entry` where a cut names its cell; unchanged where the
    /// buffer is not a pool of pages.
    pub(crate) fn with_page_entry(&self, entry: &PageEntry) -> BufferLayout {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => {
                self.with_paging(ComptimeOption::new_Some(table.with_page_entry(entry)))
            }
            ComptimeOption::None => self.clone(),
        }
    }

    /// This layout over `paging`, the rest unchanged.
    fn with_paging(&self, paging: ComptimeOption<PageTable>) -> BufferLayout {
        BufferLayout {
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            projection: comptime!(self.projection.clone()),
            rows: comptime!(self.rows),
            paging,
        }
    }

    /// `physical` located through the page table where the buffer is a pool of pages
    /// ([`PageTable::located`]), with the base the line's offset adds past them.
    fn paged(&self, physical: Coords<u32>) -> (Coords<u32>, u32) {
        #[comptime]
        match &self.paging {
            ComptimeOption::Some(table) => table.located(&physical),
            ComptimeOption::None => (physical, 0u32),
        }
    }

    /// `physical` with a swizzled stage's line `XORed` by its row's key: the line a block row's
    /// coordinates along the axes it spans name, read as one index and split back.
    fn placed(&self, physical: &Coords<u32>) -> Coords<u32> {
        match comptime!(self.rows.swizzle()) {
            Some(swizzle) => {
                let (first, span) = comptime!((swizzle.line_axis(), swizzle.span()));
                let mut line = 0u32;
                #[unroll]
                for s in 0..span {
                    line = line * comptime!(swizzle.span_lines(s) as u32) + physical.at(first + s);
                }
                let moved =
                    swizzled_line(swizzle, line, physical.at(comptime!(swizzle.row_axis())));
                let mut placed = Coords::<u32>::new();
                #[unroll]
                for pa in 0..physical.len() {
                    if pa >= first && pa < first + span {
                        let inner = comptime!(
                            (pa + 1 - first..span)
                                .map(|s| swizzle.span_lines(s))
                                .product::<usize>() as u32
                        );
                        let lines = comptime!(swizzle.span_lines(pa - first) as u32);
                        placed.push(moved / inner % lines);
                    } else {
                        placed.push(physical.at(pa));
                    }
                }
                placed
            }
            None => physical.clone(),
        }
    }
}

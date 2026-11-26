//! The runtime half of a [`Projection`] over one buffer: its shape and strides, dotted in-kernel.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::{Coords1d, CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// In-kernel twin of cubecl's `TiledViewLayout` over a physical coordinate, `projection` being
/// [`Projection::positional`]'s synthetic map.
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
    /// The order a stage lays its axes down in; the source's for every other buffer.
    #[cube(comptime)]
    pub(crate) order: StageOrder,
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
            physical.push(self.physical_coord(&pos, pa));
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
        logical_extent(&self.projection, &self.physical_shape).to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

/// A buffer layout's shape and strides where both are constants, read while its kernel expands:
/// what the offsets a copy's turns land at are worked out from before the kernel runs.
#[derive(Clone, Debug)]
pub(crate) struct ConstantLayout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) projection: Projection,
    pub(crate) rows: RowArrangement,
    pub(crate) order: StageOrder,
}

impl ConstantLayout {
    /// Where line `i` of the buffer sits, as [`BufferLayout::line_offset`] places it.
    pub(crate) fn line_offset(&self, i: usize) -> usize {
        match self.rows.is_pitched() || self.order != StageOrder::AsSource {
            true => self.dot(&self.line_physical(i)),
            false => i,
        }
    }

    /// The logical coordinates of `width`-wide line `j`, as
    /// [`BufferLayout::source_line_coords`] counts it.
    pub(crate) fn source_line(&self, j: usize, width: usize) -> Vec<usize> {
        let mut extents = self.logical_extents();
        if let Some(innermost) = extents.last_mut() {
            *innermost /= width;
        }
        unravel(j, &extents)
    }

    /// Where value `v` of the `width`-wide line at `line` sits, as [`BufferLayout::cell_offset`]
    /// places it; `None` for a buffer whose rows a swizzle permutes.
    pub(crate) fn cell_offset(&self, line: &[usize], width: usize, v: usize) -> Option<usize> {
        if self.rows.swizzle().is_some() {
            return None;
        }
        let last = line.len() - 1;
        let cell: Vec<usize> = (0..line.len())
            .map(|p| match p == last {
                true => line[p] * width + v,
                false => line[p],
            })
            .collect();
        let physical: Option<Vec<usize>> = (0..self.shape.len())
            .map(|pa| self.physical_coord(&cell, pa))
            .collect();
        Some(self.dot(&physical?))
    }

    /// The extent of each logical dimension: the product of the physical extents carrying it.
    fn logical_extents(&self) -> Vec<usize> {
        self.projection
            .dimensions()
            .iter()
            .map(|&dimension| {
                self.projection
                    .carriers(dimension)
                    .iter()
                    .map(|&pa| self.shape[pa])
                    .product()
            })
            .collect()
    }

    /// The physical coordinates line `i` sits at, row-major over the shape.
    fn line_physical(&self, i: usize) -> Vec<usize> {
        unravel(i, &self.shape)
    }

    /// The coordinate along physical axis `pa` of the logical coordinate `pos`; `None` where a
    /// dimension steps it backwards.
    fn physical_coord(&self, pos: &[usize], pa: usize) -> Option<usize> {
        self.projection
            .physical_axis(pa)
            .terms()
            .iter()
            .map(|term| {
                let p = self.projection.position(term.dimension);
                let (finer, modulo) = self.projection.carried_by(pa, term.dimension);
                let quot = pos[p] / finer.iter().map(|&f| self.shape[f]).product::<usize>();
                let coord = modulo.map_or(quot, |m| quot % self.shape[m]);
                usize::try_from(term.scale.get())
                    .ok()
                    .map(|scale| coord * scale)
            })
            .sum()
    }

    /// The offset of the physical coordinates `physical`.
    fn dot(&self, physical: &[usize]) -> usize {
        physical.iter().zip(&self.strides).map(|(c, s)| c * s).sum()
    }
}

/// The coordinates of the flat row-major index `i` over `extents`.
fn unravel(i: usize, extents: &[usize]) -> Vec<usize> {
    (0..extents.len())
        .map(|p| i / extents[p + 1..].iter().product::<usize>() % extents[p])
        .collect()
}

impl BufferLayout {
    /// This layout's shape and strides where both are constants, and the buffer is not a pool of
    /// pages.
    pub(crate) fn constants(&self) -> comptime_type!(Option<ConstantLayout>) {
        cubecl::unexpanded!()
    }
}

impl BufferLayoutExpand {
    pub(crate) fn __expand_constants_method(&self, _scope: &Scope) -> Option<ConstantLayout> {
        if !self.paging.is_none() {
            return None;
        }
        Some(ConstantLayout {
            shape: self.physical_shape.constants()?,
            strides: self.physical_strides.constants()?,
            projection: self.projection.clone(),
            rows: self.rows,
            order: self.order,
        })
    }
}

#[cube]
impl BufferLayout {
    /// The logical coordinates of `width`-wide line `j` of this buffer, counted row-major over its
    /// logical extents, the innermost in lines: the source's order, whatever order the buffer
    /// lays its axes down in. What a fill reads a source a wider line at a time by, where each
    /// line's values scatter across this buffer.
    pub(crate) fn source_line_coords(&self, j: usize, #[comptime] width: usize) -> CoordsDyn {
        let extent = logical_extent(&self.projection, &self.physical_shape);
        let rank = comptime!(self.projection.logical_rank());
        let mut lines = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(p + 1 == rank) {
                lines.push(extent.at(p).divided_by(comptime!(width as u32).runtime()));
            } else {
                lines.push(extent.at(p));
            }
        }
        lines.unravel(j.cast::<u32>()).to_dyn()
    }

    /// Where value `v` of the `width`-wide line at the logical coordinates `line` sits.
    pub(crate) fn cell_offset(
        &self,
        line: &CoordsDyn,
        #[comptime] width: usize,
        #[comptime] v: usize,
    ) -> usize {
        let rank = comptime!(self.projection.logical_rank());
        let mut cell = CoordsDyn::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(p + 1 == rank) {
                cell.push(
                    line[p]
                        .times(comptime!(width as u32).runtime())
                        .plus(comptime!(v as u32).runtime()),
                );
            } else {
                cell.push(line[p]);
            }
        }
        self.to_source_pos(cell)
    }

    /// The logical coordinate line `i` of this buffer holds.
    pub(crate) fn line_coords(&self, i: usize) -> CoordsDyn {
        let physical = self.placed(&self.line_physical(i));
        self.logical_of_physical(&physical)
    }

    /// Where line `i` of this buffer sits: `i` itself unless rows are padded or the axes are laid
    /// down in another order than the source's.
    pub(crate) fn line_offset(&self, i: usize) -> usize {
        if comptime!(self.rows.is_pitched() || self.order != StageOrder::AsSource) {
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
            order: comptime!(self.order),
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

    /// The coordinate `pos` sits at along physical axis `pa`.
    fn physical_coord(&self, pos: &CoordsDyn, #[comptime] pa: usize) -> u32 {
        let map = comptime!(self.projection.physical_axis(pa).clone());
        let picks = comptime!((0..map.terms().len()).collect::<Vec<_>>());
        let mut parts = Sequence::<u32>::new();
        #[unroll]
        for t in 0..map.terms().len() {
            let term = comptime!(map.terms()[t]);
            let p = comptime!(self.projection.position(term.dimension));
            let (finer, modulo) = comptime!(self.projection.carried_by(pa, term.dimension));
            // Strip the finer coordinates, then take this one; the outermost keeps the full
            // quotient.
            let quot = pos[p].divided_by(self.physical_shape.product(finer.to_vec()));
            let coord = match modulo {
                Some(m) => quot.remainder(self.physical_shape.at(m)),
                None => quot,
            };
            parts.push(coord.times(comptime!(term.scale.get() as u32).runtime()));
        }
        parts.sum(picks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `16 × 16` stage laid along its first Dimension, one value a line, its rows of 16 pitched
    /// to 24.
    fn along_first() -> ConstantLayout {
        let (k, n) = (Dimension(0), Dimension(1));
        ConstantLayout {
            shape: vec![16, 16],
            strides: vec![1, 24],
            projection: Projection::direct(&[k, n]),
            rows: RowArrangement::Padded { lines: 8 },
            order: StageOrder::AlongRun(k),
        }
    }

    /// A source line counted in the source's order lands each of its values a pitched row apart,
    /// and a stage line at its strided offset.
    #[test]
    fn a_constant_layout_lands_a_scattered_line_at_its_cells() {
        let layout = along_first();
        // Source line 5 of 8-wide lines over `16 × 16` is row 2, the second line along it.
        let line = layout.source_line(5, 8);
        assert_eq!(line, vec![2, 1]);
        for v in 0..8 {
            assert_eq!(layout.cell_offset(&line, 8, v), Some(2 + (8 + v) * 24));
        }
        assert_eq!(layout.line_offset(17), 1 + 24);
    }

    /// A swizzled buffer's cells are not worked out ahead.
    #[test]
    fn a_swizzled_layout_states_no_cell() {
        let layout = ConstantLayout {
            rows: RowArrangement::Swizzled(ChunkSwizzle::new(&[16, 2], LineBytes(16), 2).unwrap()),
            order: StageOrder::AsSource,
            strides: vec![2, 1],
            shape: vec![16, 2],
            ..along_first()
        };
        assert_eq!(layout.cell_offset(&[0, 0], 8, 0), None);
    }
}

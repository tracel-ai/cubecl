//! The straight transport: whole lines, each unit its own, walked in its delivery's order.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::async_copy::AsyncLines;
use super::padded::{read_stage_line, widened_shape};
use crate::*;

#[cube]
impl Memory {
    /// Straight half of [`fill_from`](Memory::fill_from); `I2`/`WP2` are the storage element and
    /// width.
    pub(crate) fn fill_straight<I2: Numeric, WP2: Size>(
        &mut self,
        src: &Memory,
        #[comptime] space: &Space,
    ) {
        // A gathered stage carries the source window's map in its own registers.
        if comptime!(self.projection.is_rational() || self.projection.has_dynamic_scales()) {
            self.map.store_from(&src.map);
        }
        let check = comptime!(src.access.overhang.masks());
        let sw = comptime!(src.store.vector_size);
        let w = comptime!(self.store.vector_size);
        // A direct stage with no source window holds its logical box, whatever the source gathers.
        let dense = comptime!(self.projection.is_direct() && self.source_window.is_none());
        let compaction = comptime!(match dense {
            true => None,
            false => stage_compaction(&src.projection, &self.projection, w, space),
        });
        comptime!(assert!(
            !dense
                || src.projection.is_direct()
                || src.projection.composition() == Composition::Disjoint
                || (sw == w && !src.projection.is_rational()),
            "Memory::fill_straight: a dense stage reads its gathered source through the source's \
             map, which a padded or rational fill does not"
        ));
        // Empty when the window has no holes to skip.
        let steps = comptime!(match &compaction {
            Some(c) if !c.is_dense() => c.steps().to_vec(),
            _ => Vec::new(),
        });
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        let total = shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>();
        let layout = self.layout.clone();
        let extent = comptime!(fill_extent(space, sw, w, check));
        let src_rank = comptime!(src.projection.physical_rank());
        let padding = comptime!((sw != w).then(|| {
            // `source_component` only addresses the source when both boxes share a rank.
            assert!(
                src_rank == plen,
                "Memory::fill_straight: a padded stage is a rank-{plen} box filled from a \
                 rank-{src_rank} source, so a destination coordinate does not address it"
            );
            Padding {
                width: w,
                extent,
                rank: src_rank,
            }
        }));
        let fill = comptime!(self.access.fill);
        let order = comptime!(
            src.access
                .delivery
                .order()
                .expect("Memory::fill_straight: a straight fill's source is delivered per unit")
        );
        let total_c = total.constant();
        // A gathered destination's line count must equal the compacted window's.
        let cells = comptime!(compaction.as_ref().map(|c| c.cells(w)));
        comptime!(assert!(
            match cells {
                Some(n) => matches!(total_c, Some(t) if t as usize == n),
                None => true,
            },
            "Memory::fill_straight: a gathered source fills a destination of {total_c:?} lines, \
             but its compacted window is {cells:?}"
        ));
        let turns = comptime!(fill.turns(total_c.map(|t| t as usize)));
        let d = self.lines_storage_mut::<I2, WP2>();
        if sw == w {
            // A partition no compaction stands in for, several of whose axes share one of its
            // physical axes, is a source of fewer physical axes than the stage's logical ones:
            // the stage's coordinates name its axes, not the source's physical ones.
            let through_projection = comptime!(
                compaction.is_none()
                    && !src.projection.is_direct()
                    && !src.projection.is_rational()
                    && (src_rank != plen || src.projection.composition() != Composition::Disjoint)
            );
            let s = if steps.is_empty() && !through_projection {
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked)),
                    check,
                )
            } else if through_projection {
                // Read at the stage's logical coordinates through the source's own projection.
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked))
                        .view(src.axis_projection(space)),
                    check,
                )
            } else {
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked))
                        .view(CompactionStep::new(shape.clone(), comptime!(steps))),
                    check,
                )
            };
            // Only an equal-width fill moves each line whole, which is what the copy engine does
            // ([`TransportKind::new`] refuses it any other).
            if comptime!(src.access.delivery == Delivery::AsyncPerUnit) {
                let mut lines = AsyncLines::<I2, WP2>::new(d, &s, &layout, order);
                FillUnits::move_lines::<AsyncLines<I2, WP2>>(fill, total, turns, &mut lines);
            } else {
                let mut lines = StraightLines::<I2, WP2, WP2>::new(d, &s, &layout, order, padding);
                FillUnits::move_lines::<StraightLines<I2, WP2, WP2>>(
                    fill, total, turns, &mut lines,
                );
            }
        } else {
            let s = if steps.is_empty() {
                Masked::new(
                    src.window_view_storage::<I2, Const<1>>(comptime!(Guard::Checked)),
                    check,
                )
            } else {
                Masked::new(
                    src.window_view_storage::<I2, Const<1>>(comptime!(Guard::Checked))
                        .view(CompactionStep::new(
                            widened_shape(&shape, comptime!(plen), comptime!(w)),
                            comptime!(steps),
                        )),
                    check,
                )
            };
            let mut lines = StraightLines::<I2, WP2, Const<1>>::new(d, &s, &layout, order, padding);
            FillUnits::move_lines::<StraightLines<I2, WP2, Const<1>>>(
                fill, total, turns, &mut lines,
            );
        }
    }
}

/// The innermost extent of `space` in cells, checked against both widths; `None` when `Dynamic`.
/// The extent must be a whole number of `sw`-wide source lines.
pub(crate) fn fill_extent(space: &Space, sw: usize, w: usize, check: bool) -> Option<usize> {
    match space.extent_raw(space.axis_at(space.rank() - 1)) {
        Extent::Static(e) => {
            assert!(
                e.is_multiple_of(sw),
                "Memory: the innermost extent {e} is not a whole number of the source's \
                 {sw}-wide lines, so the stage holds cells the source cannot hand it"
            );
            Some(e)
        }
        Extent::Dynamic => {
            assert!(
                sw == w || check,
                "Memory: a padded stage over a Dynamic innermost extent cannot know at comptime \
                 which units are padding, so its source must be bounds-checked for them to read \
                 as zero"
            );
            None
        }
    }
}

/// A straight fill's line, loaded and stored by the unit: the source's line at the coordinates
/// the line holds in the delivery's order, assembled out of scalar cells where the stage is padded
/// ([`Padding`]).
#[derive(CubeType)]
struct StraightLines<'a, I2: Numeric, WP2: Size, SW: Size> {
    d: &'a mut [Vector<I2, WP2>],
    s: &'a Masked<'a, Vector<I2, SW>, CoordsDyn>,
    layout: &'a BufferLayout,
    #[cube(comptime)]
    order: LineOrder,
    #[cube(comptime)]
    padding: Option<Padding>,
}

#[cube]
impl<'a, I2: Numeric, WP2: Size, SW: Size> StraightLines<'a, I2, WP2, SW> {
    fn new(
        d: &'a mut [Vector<I2, WP2>],
        s: &'a Masked<'a, Vector<I2, SW>, CoordsDyn>,
        layout: &'a BufferLayout,
        #[comptime] order: LineOrder,
        #[comptime] padding: Option<Padding>,
    ) -> Self {
        StraightLines::<'a, I2, WP2, SW> {
            d,
            s,
            layout,
            order,
            padding,
        }
    }
}

#[cube]
impl<'a, I2: Numeric, WP2: Size, SW: Size> LineMove for StraightLines<'a, I2, WP2, SW> {
    fn move_line(&mut self, line: usize, #[comptime] _turn: Option<usize>) {
        let (coords, offset) = self.layout.line_in(line, comptime!(self.order));
        self.d[offset] = read_stage_line::<I2, WP2, SW>(self.s, &coords, comptime!(self.padding));
    }
}

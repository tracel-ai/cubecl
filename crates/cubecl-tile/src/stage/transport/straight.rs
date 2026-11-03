//! The straight transport: the destination filled in its own physical order, whole lines.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::async_copy::fill_lines_async;
use super::padded::{read_stage_line, widened_shape};
use crate::*;

#[cube]
impl Memory {
    /// Straight half of [`fill_from`](Memory::fill_from); `I2`/`WP2` are the storage element and
    /// width.
    pub(crate) fn fill_straight<I2: Numeric, WP2: Size>(
        &mut self,
        src: &Memory,
        #[comptime] space: Space,
    ) {
        // A gathered stage carries the source window's map in its own registers.
        if comptime!(self.projection.is_rational() || self.projection.has_dynamic_scales()) {
            self.map.store_from(&src.map);
        }
        let check = comptime!(src.access.overhang.masks());
        let sw = comptime!(src.store.vector_size);
        let w = comptime!(self.store.vector_size);
        let compaction = comptime!(stage_compaction(
            &src.projection,
            &self.projection,
            w,
            &space
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
        let extent = comptime!(fill_extent(&space, sw, w, check));
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
        // A comptime worker count unrolls the tasks: a runtime stride (`CUBE_DIM`, or a plane's
        // `CUBE_DIM_X`) stalls each store on Metal's in-order pipe. Unknown or tiny cubes stay rolled.
        let fill = comptime!(self.access.fill);
        let units = comptime!(fill.count);
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
        let straight =
            comptime!(matches!(total_c, Some(t) if units > 0 && (t as usize).div_ceil(units) <= 8));
        let d = self.lines_storage_mut::<I2, WP2>();
        if comptime!(sw == w) {
            // A partition no compaction stands in for, several of whose axes share one of its
            // physical axes, is a source of fewer physical axes than the stage's logical ones:
            // the stage's coordinates name its axes, not the source's physical ones.
            let through_projection = comptime!(
                compaction.is_none()
                    && !src.projection.is_direct()
                    && !src.projection.is_rational()
                    && src_rank != plen
            );
            let s = if comptime!(steps.is_empty() && !through_projection) {
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked)),
                    check,
                )
            } else if comptime!(through_projection) {
                // Read at the stage's logical coordinates through the source's own projection.
                Masked::new(
                    src.window_view_storage::<I2, WP2>(comptime!(Guard::Checked))
                        .view(src.axis_projection(comptime!(space.clone()))),
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
                fill_lines_async::<I2, WP2>(d, &s, &layout, total, total_c, fill, straight);
            } else {
                fill_lines::<I2, WP2, WP2>(d, &s, &layout, total, total_c, fill, straight, padding);
            }
        } else {
            let s = if comptime!(steps.is_empty()) {
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
            fill_lines::<I2, WP2, Const<1>>(
                d, &s, &layout, total, total_c, fill, straight, padding,
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

/// Cooperatively write destination stage lines cyclically across the units `fill` names.
#[cube]
fn fill_lines<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    layout: &BufferLayout,
    total: usize,
    #[comptime] total_c: Option<u64>,
    #[comptime] fill: FillUnits,
    #[comptime] straight: bool,
    #[comptime] padding: Option<Padding>,
) {
    if comptime!(straight) {
        let units = comptime!(fill.count);
        let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
        #[unroll]
        for t in 0..tasks {
            let i = FillUnits::worker(fill) + comptime!(t * units);
            if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                if i < total {
                    d[layout.line_offset(i)] = read_stage_line::<I2, WP2, SW>(
                        s,
                        &layout.line_coords(i),
                        comptime!(padding),
                    );
                }
            } else {
                d[layout.line_offset(i)] =
                    read_stage_line::<I2, WP2, SW>(s, &layout.line_coords(i), comptime!(padding));
            }
        }
    } else {
        let stride = FillUnits::workers(fill);
        let mut i = FillUnits::worker(fill);
        while i < total {
            d[layout.line_offset(i)] =
                read_stage_line::<I2, WP2, SW>(s, &layout.line_coords(i), comptime!(padding));
            i += stride;
        }
    }
}

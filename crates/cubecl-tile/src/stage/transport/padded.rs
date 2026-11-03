//! Assembling one destination line out of scalar source cells, for a padded stage.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// What a padded fill needs beyond the two boxes: source cells per line and the padding extent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) width: usize,
    pub(crate) extent: Option<usize>,
    /// The physical rank both boxes share.
    pub(crate) rank: usize,
}

/// Read one destination line at `pos`, whole or assembled from scalar cells ([`widen_line`]).
#[cube]
pub(crate) fn read_stage_line<I2: Numeric, WP2: Size, SW: Size>(
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Option<Padding>,
) -> Vector<I2, WP2> {
    if comptime!(padding.is_some()) {
        widen_line::<I2, WP2, SW>(s, pos, comptime!(padding.unwrap()))
    } else {
        // `SW` is `WP2` here; the cast is an identity.
        Vector::<I2, WP2>::cast_from(s.read(pos.clone()))
    }
}

/// Assemble one padded destination line from adjacent scalar source cells.
/// With `Padding::extent` `None`, the source must be bounds-checked so padding reads zero.
#[cube]
fn widen_line<T: Numeric, W: Size, SW: Size>(
    s: &Masked<'_, Vector<T, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Padding,
) -> Vector<T, W> {
    let width = comptime!(padding.width);
    let rank = comptime!(padding.rank);
    comptime!(assert!(
        SW::try_value_const() == Some(1),
        "widen_line: a padded stage is filled from a scalar source, got a {:?}-wide one",
        SW::try_value_const()
    ));
    comptime!(assert!(
        W::try_value_const().is_none_or(|n| n == width),
        "widen_line: assembles {width} units into a {:?}-wide destination line",
        W::try_value_const()
    ));
    let last = comptime!(rank - 1);
    let line = pos[last];
    let mut out = Vector::<T, W>::cast_from(T::from_int(0));
    let guarded = comptime!(match padding.extent {
        Some(n) => !n.is_multiple_of(width),
        None => false,
    });
    #[unroll]
    for l in 0..width {
        let cell = line
            .times(comptime!(width as u32))
            .plus(comptime!(l as u32));
        let valid = if comptime!(guarded) {
            cell < comptime!(padding.extent.unwrap() as u32)
        } else {
            true.runtime()
        };
        if valid {
            out.insert(
                l,
                s.read(source_component(pos, comptime!(rank), cell))
                    .extract(0usize),
            );
        }
    }
    out
}

/// Replace the destination line coordinate with its scalar source-cell coordinate.
#[cube]
fn source_component(pos: &CoordsDyn, #[comptime] rank: usize, cell: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(cell);
        } else {
            out.push(pos[p]);
        }
    }
    out
}

/// Express a padded destination's innermost physical extent in scalar source elements.
#[cube]
pub(crate) fn widened_shape(
    shape: &Coords<u32>,
    #[comptime] rank: usize,
    #[comptime] width: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(shape.at(p).times(comptime!(width as u32)));
        } else {
            out.push(shape.at(p));
        }
    }
    out
}

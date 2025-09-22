use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::batch::partitioned_matmul::hypercube::base::CubeSpan;

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Describes the global traversal order as flattened cube position increases.
///
/// - `RowMajor`: standard row-first traversal
/// - `ColMajor`: standard column-first traversal
/// - `SwizzleColMajor(w)`: zigzag pattern down columns, with `w`-wide steps
/// - `SwizzleRowMajor(w)`: zigzag pattern across rows, with `w`-wide steps
///
/// Special cases:
/// - `SwizzleColMajor(1)` is equivalent to `ColMajor`
/// - `SwizzleRowMajor(1)` is equivalent to `RowMajor`
#[allow(clippy::enum_variant_names)]
pub enum GlobalOrder {
    #[default]
    RowMajor,
    ColMajor,
    SwizzleRowMajor(u32),
    SwizzleColMajor(u32),
}

impl GlobalOrder {
    /// Since they are equivalent but the latter form will skip some calculations,
    /// - `SwizzleColMajor(1)` becomes `ColMajor`
    /// - `SwizzleRowMajor(1)` becomes `RowMajor`
    pub fn canonicalize(self) -> Self {
        match self {
            GlobalOrder::SwizzleColMajor(1) => GlobalOrder::ColMajor,
            GlobalOrder::SwizzleRowMajor(1) => GlobalOrder::RowMajor,
            _ => self,
        }
    }
}

#[derive(Default)]
/// Used to create [GlobalOrder].
#[allow(unused)]
pub enum GlobalOrderSelection {
    /// It creates the default global order.
    #[default]
    Default,
    /// Set a global order.
    Fixed(GlobalOrder),
    /// Creates swizzle row global order if possible.
    ///
    /// Fallbacks to row global order otherwise.
    SwizzleRow { m: u32, w: u32 },
    /// Creates swizzle col global order if possible.
    ///
    /// Fallbacks to col global order otherwise.
    SwizzleCol { n: u32, w: u32 },
}

impl GlobalOrderSelection {
    pub fn into_order(self, span: &CubeSpan) -> GlobalOrder {
        match self {
            GlobalOrderSelection::Default => GlobalOrder::default(),
            GlobalOrderSelection::Fixed(order) => order,
            GlobalOrderSelection::SwizzleRow { m, w } => {
                let m_cubes = m.div_ceil(span.m);
                if m_cubes % w != 0 {
                    GlobalOrder::RowMajor
                } else {
                    GlobalOrder::SwizzleRowMajor(w)
                }
            }
            GlobalOrderSelection::SwizzleCol { n, w } => {
                let n_cubes = n.div_ceil(span.n);
                if n_cubes % w != 0 {
                    GlobalOrder::RowMajor
                } else {
                    GlobalOrder::SwizzleRowMajor(w)
                }
            }
        }
        .canonicalize()
    }
}

#[cube]
/// Maps a linear `index` to 2D zigzag coordinates `(x, y)` within horizontal or vertical strips.
///
/// Each strip is made of `num_steps` steps, each of length `step_length`.
/// Strips alternate direction: even strips go top-down, odd strips bottom-up.
/// Steps alternate direction: even steps go left-to-right, odd steps right-to-left.
///
/// - Prefer **odd `num_steps`** for smoother transitions between strips.
/// - Prefer **power-of-two `step_length`** for better performance.
///
/// # Parameters
/// - `index`: linear input index
/// - `num_steps`: number of snaking steps in a strip
/// - `step_length`: number of elements in each step (must be > 0)
///
/// # Returns
/// `(x, y)` coordinates after swizzling
pub fn swizzle(index: u32, num_steps: u32, #[comptime] step_length: u32) -> Coords2d {
    comptime!(assert!(step_length > 0));

    let num_elements_per_strip = num_steps * step_length;
    let strip_index = index / num_elements_per_strip;
    let pos_in_strip = index % num_elements_per_strip;
    let strip_offset = step_length * strip_index;

    // Indices without regards to direction
    let abs_step_index = pos_in_strip / step_length;
    let abs_pos_in_step = pos_in_strip % step_length;

    // Top-down (0) or Bottom-up (1)
    let strip_direction = strip_index % 2;
    // Left-right (0) or Right-left (1)
    let step_direction = abs_step_index % 2;

    // Update indices with direction
    let step_index =
        strip_direction * (num_steps - abs_step_index - 1) + (1 - strip_direction) * abs_step_index;

    let pos_in_step = if comptime!(step_length & (step_length - 1) == 0) {
        abs_pos_in_step ^ (step_direction * (step_length - 1))
    } else {
        step_direction * (step_length - abs_pos_in_step - 1)
            + (1 - step_direction) * abs_pos_in_step
    };

    (step_index, pos_in_step + strip_offset)
}

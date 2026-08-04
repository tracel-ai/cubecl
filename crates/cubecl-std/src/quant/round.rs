use cubecl::prelude::*;
use cubecl_common::quant::scheme::{F32Grid, QuantParam};
use cubecl_core as cubecl;

/// The smallest value representable in `param` that is not below `scale`, in a kernel.
///
/// Device-side counterpart of [`QuantParam::round_up`], and the two have to agree: a tensor
/// quantized on one backend has to reconstruct the same on another.
///
/// Returned as `F` rather than the storage type because the result is exactly representable in
/// `param`, so the caller's cast to it is lossless.
///
/// `F` only carries the value in and out. The rule runs in f32, so a narrow `F` cannot turn the
/// saturation bound into an infinity or the subnormal spacing into a flushed zero.
///
/// `scale` must not be negative, as with the host rule.
#[cube]
pub fn round_up_to_param<F: Float>(scale: F, #[comptime] param: QuantParam) -> F {
    #[comptime]
    match param {
        QuantParam::F32 => scale,
        QuantParam::F16 | QuantParam::BF16 | QuantParam::UE4M3 => {
            F::cast_from(step_up(f32::cast_from(scale), param))
        }
        // Returning `scale` would diverge from the host rule, which has no answer here either.
        QuantParam::UE8M0 => comptime!(unimplemented!("UE8M0 scales are not yet supported")),
    }
}

#[cube]
fn step_up(scale: f32, #[comptime] param: QuantParam) -> f32 {
    // Mirrors QuantParam::round_up, saturating at the top rather than converting past it: above the
    // maximum a conversion gives an infinity, and every value scaled by it then reconstructs wrong.
    // Both paths below work on the f32 bit pattern rather than the storage type, because the
    // narrowing conversion that would replace them is one the WGSL path leaves unrounded.
    let grid = comptime!(param.f32_grid());
    let max = comptime!(param.max_representable());

    if scale >= max {
        max
    } else if comptime!(grid.subnormals.is_some()) {
        let subnormals = comptime!(grid.subnormals.unwrap());
        let spacing = comptime!(subnormals.spacing);

        // Below the minimum normal the spacing stops halving, so the answer is a count of steps.
        if scale < comptime!(subnormals.min_normal) {
            f32::ceil(scale / spacing) * spacing
        } else {
            round_up_on_grid(scale, grid)
        }
    } else {
        round_up_on_grid(scale, grid)
    }
}

/// Rounds `scale` up onto `grid`, for a value in the param's normal range and below its maximum.
///
/// Truncating the low f32 mantissa bits lands on the grid, and biasing first turns that truncation
/// into a round up.
#[cube]
fn round_up_on_grid(scale: f32, #[comptime] grid: F32Grid) -> f32 {
    let bits = u32::reinterpret(scale);
    let up_bits = (bits + comptime!(grid.round_up_bias())) & comptime!(grid.truncate_mask());
    f32::reinterpret(up_bits)
}

use cubecl::prelude::*;
use cubecl_common::quant::scheme::{F32Grid, ScaleDtype};
use cubecl_core as cubecl;

/// The smallest value representable in `dtype` that is not below `scale`, in a kernel.
///
/// Device-side counterpart of [`ScaleDtype::round_up`], and the two have to agree: a tensor
/// quantized on one backend has to reconstruct the same on another.
///
/// Returned as `F` rather than the storage type because the result is exactly representable in
/// `dtype`, so the caller's cast to it is lossless.
///
/// `F` only carries the value in and out. The rule runs in f32, so a narrow `F` cannot turn the
/// saturation bound into an infinity or the subnormal spacing into a flushed zero.
///
/// `scale` must not be negative, as with the host rule.
#[cube]
pub fn round_up_to_dtype<F: Float>(scale: F, #[comptime] dtype: ScaleDtype) -> F {
    #[comptime]
    match dtype {
        ScaleDtype::F32 => scale,
        ScaleDtype::F16 | ScaleDtype::BF16 | ScaleDtype::UE4M3 => {
            F::cast_from(step_up(f32::cast_from(scale), dtype))
        }
        ScaleDtype::UE8M0 => F::cast_from(step_up_to_power_of_two(f32::cast_from(scale))),
    }
}

/// The `ue8m0` arm of [`round_up_to_dtype`]: the smallest power of two not below `scale`.
///
/// Kept apart from [`step_up`] because both of `ue8m0`'s ends need clamping before the shared
/// stepping means anything — its bottom, 2^-127, is subnormal in f32, and it has no zero for a
/// fully-zero block to calibrate to. Mirrors the host's `round_up_to_power_of_two`, and the two
/// have to keep agreeing or a tensor quantized on one backend reconstructs differently on another.
#[cube]
fn step_up_to_power_of_two(scale: f32) -> f32 {
    let min = comptime!(ScaleDtype::UE8M0_MIN);
    let max = comptime!(ScaleDtype::UE8M0_MAX);

    if scale <= min {
        min
    } else if scale >= max {
        max
    } else {
        round_up_on_grid(scale, comptime!(ScaleDtype::UE8M0.f32_grid()))
    }
}

#[cube]
fn step_up(scale: f32, #[comptime] dtype: ScaleDtype) -> f32 {
    // Mirrors ScaleDtype::round_up, saturating at the top rather than converting past it: above the
    // maximum a conversion gives an infinity, and every value scaled by it then reconstructs wrong.
    // Both paths below work on the f32 bit pattern rather than the storage type, because the
    // narrowing conversion that would replace them is one the WGSL path leaves unrounded.
    let grid = comptime!(dtype.f32_grid());
    let max = comptime!(dtype.max_representable());

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

/// Rounds `scale` up onto `grid`, for a value in the dtype's normal range and below its maximum.
///
/// Truncating the low f32 mantissa bits lands on the grid, and biasing first turns that truncation
/// into a round up.
#[cube]
fn round_up_on_grid(scale: f32, #[comptime] grid: F32Grid) -> f32 {
    let bits = u32::reinterpret(scale);
    let up_bits = (bits + comptime!(grid.round_up_bias())) & comptime!(grid.truncate_mask());
    f32::reinterpret(up_bits)
}

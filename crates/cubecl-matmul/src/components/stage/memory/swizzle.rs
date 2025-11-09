use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::Swizzle;

use crate::components::stage::SwizzleMode;

#[cube]
pub fn as_swizzle_object(#[comptime] mode: SwizzleMode) -> Swizzle {
    let bits = comptime![match mode {
        SwizzleMode::None => 0u32,
        SwizzleMode::B32 => 1,
        SwizzleMode::B64 => 2,
        SwizzleMode::B128 => 3,
    }];
    Swizzle::new(bits, 4u32, 3)
}

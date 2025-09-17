use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use variadics_please::all_tuples_with_size;

/// A set of coordinates used in layouts. Contains some utilities for comptime inspection.
#[cube]
pub trait Coordinates: CubeType + Clone {
    /// Add two coordinates together and return the result.
    fn add(this: Self, other: Self) -> Self;
    /// Subtract two coordinates from each other and return the result.
    fn sub(this: Self, other: Self) -> Self;
    /// Apply an elementwise minimum to the coordinates and return the result.
    fn min(this: Self, other: Self) -> Self;
    /// Apply an elementwise maximum to the coordinates and return the result.
    fn max(this: Self, other: Self) -> Self;
    /// Check whether `pos` is fully contained within `bounds`.
    fn is_in_bounds(pos: Self, bounds: Self) -> bool;
    /// The origin (zero) coordinates. `this` may be used as a reference coordinate for dynamically
    /// sized layouts.
    fn origin(this: &Self) -> Self;
}

// Aliases for convenience and semantic clarity
pub type Coords1d = u32;
pub type Coords2d = (u32, u32);
pub type Coords3d = (u32, u32, u32);
pub type Coords4d = (u32, u32, u32, u32);
pub type Coords5d = (u32, u32, u32, u32, u32);
pub type CoordsDyn = Sequence<u32>;

macro_rules! repeat_tokens {
    ($t:ident, $tokens:tt) => {
        $tokens
    };
}

macro_rules! impl_coordinates_tuple {
    ($N:expr, $(($t:ident, $o: ident)),*) => {
        paste::paste! {
            mod [<coords $N d>] {
                use super::*;
                const ORIGIN: ($(repeat_tokens!($t, u32),)*) = ($(repeat_tokens!($t, 0u32),)*);

                #[cube]
                impl Coordinates for ($(repeat_tokens!($t, u32),)*) {
                    fn add(this: Self, other: Self) -> Self {
                        let ($($t),*) = this;
                        let ($($o),*) = other;
                        ($($t + $o),*)
                    }
                    fn sub(this: Self, other: Self) -> Self {
                        let ($($t),*) = this;
                        let ($($o),*) = other;
                        ($($t - $o),*)
                    }
                    fn min(this: Self, other: Self) -> Self {
                        let ($($t),*) = this;
                        let ($($o),*) = other;
                        ($(Min::min($t, $o)),*)
                    }
                    fn max(this: Self, other: Self) -> Self {
                        let ($($t),*) = this;
                        let ($($o),*) = other;
                        ($(Max::max($t, $o)),*)
                    }
                    fn is_in_bounds(this: Self, other: Self) -> bool {
                        let ($($t),*) = this;
                        let ($($o),*) = other;
                        true $(&& $t < $o)*
                    }
                    fn origin(_this: &Self) -> Self {
                        ORIGIN.runtime()
                    }
                }
            }
        }
    };
}

#[cube]
impl Coordinates for Coords1d {
    fn add(this: Self, other: Self) -> Self {
        this + other
    }
    fn sub(this: Self, other: Self) -> Self {
        Max::max(this as i32 - other as i32, 0) as u32
    }
    fn min(this: Self, other: Self) -> Self {
        Min::min(this, other)
    }
    fn max(this: Self, other: Self) -> Self {
        Max::max(this, other)
    }
    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        pos < bounds
    }
    fn origin(_this: &Self) -> Self {
        0u32.runtime()
    }
}

all_tuples_with_size!(impl_coordinates_tuple, 2, 5, t, o);

#[cube]
impl Coordinates for CoordsDyn {
    fn add(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(*this.index(i) + *other.index(i));
        }

        out
    }

    fn sub(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(*this.index(i) - *other.index(i));
        }

        out
    }

    fn min(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(Min::min(*this.index(i), *other.index(i)));
        }

        out
    }

    fn max(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(Max::max(*this.index(i), *other.index(i)));
        }

        out
    }

    fn is_in_bounds(pos: Self, bounds: Self) -> bool {
        let rank = comptime![pos.len()];
        let mut out = true;

        #[unroll]
        for i in 0..rank {
            out &= *pos.index(i) < *bounds.index(i);
        }

        out
    }

    fn origin(this: &Self) -> Self {
        let rank = comptime![this.len()];
        let mut origin = Sequence::new();

        #[unroll]
        for _ in 0..rank {
            origin.push(0u32.runtime())
        }

        origin
    }
}

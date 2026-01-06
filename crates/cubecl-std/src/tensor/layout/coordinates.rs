use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use variadics_please::all_tuples;

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
    fn is_in_bounds(pos: &Self, bounds: &Self) -> bool;
    /// Create a new coordinates object where all values are `value`.
    /// `this` may be used as a reference coordinate for dynamically sized layouts.
    fn from_int(this: &Self, #[comptime] value: i64) -> Self;
}

// Aliases for convenience and semantic clarity
pub type Coords1d = u32;
pub type Coords1i = i32;
pub type Coords2d = (u32, u32);
pub type Coords2i = (i32, i32);
pub type Coords3d = (u32, u32, u32);
pub type Coords3i = (i32, i32, i32);
pub type Coords4d = (u32, u32, u32, u32);
pub type Coords4i = (i32, i32, i32, i32);
pub type Coords5d = (u32, u32, u32, u32, u32);
pub type Coords5i = (i32, i32, i32, i32, i32);
pub type CoordsDyn = Sequence<u32>;

macro_rules! impl_coordinates_tuple {
    ($(($T:ident, $t:ident, $o: ident)),*) => {
        // Need to force off debug symbols because of macro hygiene weirdness.
        #[cube(no_debug_symbols)]
        impl<$($T: Coordinates),*> Coordinates for ($($T),*) {
            fn add(this: Self, other: Self) -> Self {
                let ($($t),*) = this;
                let ($($o),*) = other;
                ($($T::add($t, $o)),*)
            }
            fn sub(this: Self, other: Self) -> Self {
                let ($($t),*) = this;
                let ($($o),*) = other;
                ($($T::sub($t, $o)),*)
            }
            fn min(this: Self, other: Self) -> Self {
                let ($($t),*) = this;
                let ($($o),*) = other;
                ($($T::min($t, $o)),*)
            }
            fn max(this: Self, other: Self) -> Self {
                let ($($t),*) = this;
                let ($($o),*) = other;
                ($($T::max($t, $o)),*)
            }
            fn is_in_bounds(this: &Self, other: &Self) -> bool {
                let ($($t),*) = this;
                let ($($o),*) = other;
                true $(&& $T::is_in_bounds($t, $o))*
            }
            fn from_int(this: &Self, #[comptime] value: i64) -> Self {
                let ($($t),*) = this;
                ($($T::from_int($t, value)),*)
            }
        }
    };
}

// Can't blanket implement because of trait rules
macro_rules! impl_coordinates_primitive {
    ($ty: ty) => {
        #[cube]
        impl Coordinates for $ty {
            fn add(this: Self, other: Self) -> Self {
                this + other
            }
            fn sub(this: Self, other: Self) -> Self {
                this - other
            }
            fn min(this: Self, other: Self) -> Self {
                this.min(other)
            }
            fn max(this: Self, other: Self) -> Self {
                this.max(other)
            }
            fn is_in_bounds(pos: &Self, bounds: &Self) -> bool {
                pos < bounds
            }
            fn from_int(_this: &Self, #[comptime] value: i64) -> Self {
                <$ty as Numeric>::from_int(value)
            }
        }
    };
    ($($ty: ty),*) => {
        $(impl_coordinates_primitive!($ty);)*
    }
}

impl_coordinates_primitive!(u8, u16, u32, u64, i8, i16, i32, i64);
all_tuples!(impl_coordinates_tuple, 2, 12, T, t, o);

#[cube]
impl<T: Coordinates + Copy> Coordinates for Sequence<T> {
    fn add(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(T::add(*this.index(i), *other.index(i)));
        }

        out
    }

    fn sub(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(T::sub(*this.index(i), *other.index(i)));
        }

        out
    }

    fn min(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(T::min(*this.index(i), *other.index(i)));
        }

        out
    }

    fn max(this: Self, other: Self) -> Self {
        let rank = comptime![this.len()];
        let mut out = Sequence::new();

        #[unroll]
        for i in 0..rank {
            out.push(T::max(*this.index(i), *other.index(i)));
        }

        out
    }

    fn is_in_bounds(pos: &Self, bounds: &Self) -> bool {
        let rank = comptime![pos.len()];
        let mut out = true;

        #[unroll]
        for i in 0..rank {
            out &= T::is_in_bounds(pos.index(i), bounds.index(i));
        }

        out
    }

    fn from_int(this: &Self, #[comptime] value: i64) -> Self {
        let rank = comptime![this.len()];
        let mut origin = Sequence::new();

        #[unroll]
        for i in 0..rank {
            origin.push(T::from_int(this.index(i), value));
        }

        origin
    }
}

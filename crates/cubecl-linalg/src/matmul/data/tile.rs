use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;

#[cube]
pub trait Tile<'a, E: CubePrimitive>: CubeType {
    const X: u32;
    const Y: u32;

    fn new(slice: &'a Slice<'a, E>, layout: MatrixLayout) -> Self;
    fn as_slice(tile: &Self) -> &Slice<'_, E>;
}

#[derive(CubeType)]
pub struct Tmp<'a, E: CubePrimitive> {
    slice: &'a Slice<'a, E>,
    layout: MatrixLayout,
}

macro_rules! define_tile {
    ($name:ident, $x:expr, $y:expr) => {
        #[derive(CubeType)]
        pub struct $name<'a, E: CubePrimitive> {
            slice: &'a Slice<'a, E>,
            layout: MatrixLayout,
        }

        #[cube]
        impl<'a, E: CubePrimitive> Tile<'a, E> for $name<'a, E> {
            const X: u32 = $x;
            const Y: u32 = $y;

            fn new(slice: &'a Slice<'a, E>, layout: MatrixLayout) -> Self {
                $name::<'a, E> { slice, layout }
            }

            fn as_slice(tile: &Self) -> &Slice<'_, E> {
                &tile.slice
            }
        }
    };
}

define_tile!(Tile16x16, 16, 16);
define_tile!(Tile32x8, 32, 8);
define_tile!(Tile8x32, 8, 32);

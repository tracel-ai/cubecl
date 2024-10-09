use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::tiled_layout::TilingOrder;
use crate::matmul::tile_io::loading::LhsTensorLoader;
use crate::matmul::tile_io::Loader;

use super::Tile;

#[cube]
/// Blocks are created with a filled shared memory,
/// then can give out tiles
/// When shared memory changes, a new block should be made
///
/// Double buffering: we could have two blocks, sharing a SharedMemory
pub trait Block<'a, E: CubePrimitive, T: Tile<'a, E>>: CubeType {
    const NUM_X: u32;
    const NUM_Y: u32;
    /// The goal is to encapsulate loading logic elsewhere
    type Loader: Loader<E>;

    fn new(layout: MatrixLayout) -> Self;
    fn fill(block: &'a Self, gmem_loader: Self::Loader);
    fn get_tile(block: &'a Self, x: u32, y: u32) -> T;
}

#[derive(CubeType)]
pub struct TmpBlock<E: CubePrimitive, O: TilingOrder> {
    smem: SharedMemory<E>,
    layout: MatrixLayout,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T> for TmpBlock<E, O> {
    const NUM_X: u32 = 2;
    const NUM_Y: u32 = 2;
    // TODO not specific to Lhs
    // TODO E not line
    type Loader = LhsTensorLoader<E, O>;

    fn new(layout: MatrixLayout) -> Self {
        // MAKE NEW SHARED MEMORY
        TmpBlock::<E, O> {
            smem,
            layout,
            _tiling_order: PhantomData::<O>.runtime(),
        }
    }

    fn fill(block: &'a Self, gmem_loader: Self::Loader) {
        let x = Self::Loader::load_block(gmem_loader);
    }

    fn get_tile(block: &'a Self, x: u32, y: u32) -> T {
        let tile_stride = T::X * T::Y;

        let start = O::to_nth_tile(x, y, 2, 2) * tile_stride;

        T::new(block.smem.slice(start, start + tile_stride), block.layout)
    }
}

macro_rules! define_block_smem {
    ($block_name:ident, $num_x:expr, $num_y:expr) => {
        #[derive(CubeType)]
        pub struct $block_name<E: CubePrimitive, O: TilingOrder> {
            smem: SharedMemory<E>,
            layout: MatrixLayout,
            _tiling_order: PhantomData<O>,
        }

        #[cube]
        impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T>
            for $block_name<E, O>
        {
            const NUM_X: u32 = $num_x;
            const NUM_Y: u32 = $num_y;
            type Smem = SharedMemory<E>;

            fn new_filled(smem: Self::Smem, layout: MatrixLayout) -> Self {
                $block_name::<E, O> {
                    smem,
                    layout,
                    _tiling_order: PhantomData::<O>.runtime(),
                }
            }

            fn get_tile(block: &'a Self, x: u32, y: u32) -> T {
                let tile_stride = T::X * T::Y;

                // TODO should be <Self as Block<'a, E, T>>::NUM_X / NUM_Y
                // instead of $num_x / $num_y, but cube doesnt parse it well

                // TODO block.layout influence?
                let start = O::to_nth_tile(x, y, $num_x, $num_y) * tile_stride;

                T::new(block.smem.slice(start, start + tile_stride), block.layout)
            }
        }
    };
}

define_block_smem!(Block1x1Smem, 1, 1);
define_block_smem!(Block2x1Smem, 2, 1);
define_block_smem!(Block1x2Smem, 1, 2);
define_block_smem!(Block2x2Smem, 2, 2);

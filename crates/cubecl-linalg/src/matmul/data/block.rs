use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::loading::tiled_layout::TilingOrder;

use super::Tile;

#[cube]
pub trait Block<'a, E: CubePrimitive, T: Tile<'a, E>>: CubeType {
    const NUM_X: u32;
    const NUM_Y: u32;

    fn get_tile(block: &'a Self, x: u32, y: u32) -> T;
}

#[derive(CubeType)]
pub struct Block1x1Smem<E: CubePrimitive, O: TilingOrder> {
    smem: SharedMemory<E>,
    layout: MatrixLayout,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T> for Block1x1Smem<E, O> {
    const NUM_X: u32 = 1;
    const NUM_Y: u32 = 1;

    fn get_tile(block: &'a Self, _x: u32, _y: u32) -> T {
        let start = 0;
        let tile_stride = T::X * T::Y;
        T::new(block.smem.slice(start, start + tile_stride), block.layout)
    }
}

#[derive(CubeType)]
pub struct Block2x1Smem<E: CubePrimitive, O: TilingOrder> {
    smem: SharedMemory<E>,
    layout: MatrixLayout,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T> for Block2x1Smem<E, O> {
    const NUM_X: u32 = 2;
    const NUM_Y: u32 = 1;

    fn get_tile(block: &'a Self, x: u32, _y: u32) -> T {
        let tile_stride = T::X * T::Y;
        let start = x * tile_stride;
        T::new(block.smem.slice(start, start + tile_stride), block.layout)
    }
}

#[derive(CubeType)]
pub struct Block1x2Smem<E: CubePrimitive, O: TilingOrder> {
    smem: SharedMemory<E>,
    layout: MatrixLayout,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T> for Block1x2Smem<E, O> {
    const NUM_X: u32 = 1;
    const NUM_Y: u32 = 2;

    fn get_tile(block: &'a Self, _x: u32, y: u32) -> T {
        let tile_stride = T::X * T::Y;
        let start = y * tile_stride;
        T::new(block.smem.slice(start, start + tile_stride), block.layout)
    }
}

#[derive(CubeType)]
pub struct Block2x2Smem<E: CubePrimitive, O: TilingOrder> {
    smem: SharedMemory<E>,
    layout: MatrixLayout,
    _tiling_order: PhantomData<O>,
}

#[cube]
impl<'a, E: CubePrimitive, T: Tile<'a, E>, O: TilingOrder> Block<'a, E, T> for Block2x2Smem<E, O> {
    const NUM_X: u32 = 2;
    const NUM_Y: u32 = 2;

    fn get_tile(block: &'a Self, x: u32, y: u32) -> T {
        let tile_stride = T::X * T::Y;
        // TODO should be <Self as Block<'a, E, T>>::NUM_X or NUM_Y instead of 2,
        // but cube doesnt parse it well
        let start = O::to_nth_tile(x, y, 2, 2) * tile_stride;
        T::new(block.smem.slice(start, start + tile_stride), block.layout)
    }
}

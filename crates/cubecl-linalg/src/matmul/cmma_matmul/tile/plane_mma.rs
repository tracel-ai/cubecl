use crate::matmul::cmma_matmul::tile::as_dummy_layout;
use crate::matmul::cmma_matmul::tile::DummyLayout;
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matmul_tile::TmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::matrix::MatrixLayout;
use crate::matmul::Matmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::OwnedTile;

pub struct PlaneMma32x32x32<I: Numeric, O: Numeric, T: TmmConfig> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
    _config: PhantomData<T>,
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig> PlaneMapper for PlaneMma32x32x32<I, O, T> {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
impl<I: Numeric, O: Numeric, T: TmmConfig> TileMatmul<I, O, T> for PlaneMma32x32x32<I, O, T> {
    const M: u32 = 32;
    const N: u32 = 32;
    const K: u32 = 16;

    type Lhs = OwnedTile<I>;
    type Rhs = OwnedTile<I>;
    type Out = OwnedTile<O>;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
        for k in 0..Self::K {
            let a_pk = lhs.handle[k];
            let b_kp = rhs.handle[k];

            #[unroll]
            for n in 0..Self::N {
                let b_kn = subcube_broadcast::<I>(b_kp, n);
                out.handle[n] += O::cast_from(a_pk * b_kn);
            }
        }
    }

    fn init_lhs(#[comptime] config: T) -> Self::Lhs {
        OwnedTile::<I> {
            x: Self::K,
            y: 1,
            handle: Array::new(Self::K),
            layout: as_dummy_layout(config.layout(Ident::Lhs)),
        }
    }

    fn init_rhs(#[comptime] config: T) -> Self::Rhs {
        OwnedTile::<I> {
            x: 1,
            y: Self::K,
            handle: Array::new(Self::K),
            layout: as_dummy_layout(config.layout(Ident::Rhs)),
        }
    }

    fn fill_lhs(slice: &Slice<'_, Line<I>>, lhs: &mut Self::Lhs) {
        match comptime!(lhs.layout) {
            DummyLayout::RowMajor => fill_parallel(
                slice,
                lhs.handle.as_slice_mut(),
                Self::plane_unit(),
                Self::K,
                Line::size(&slice[0]),
            ),
            DummyLayout::ColMajor => fill_perpendicular(
                slice,
                lhs.handle.as_slice_mut(),
                Self::plane_unit(),
                Self::M,
                Self::K,
                Line::size(&slice[0]),
            ),
        }
    }

    fn fill_rhs(slice: &Slice<'_, Line<I>>, rhs: &mut Self::Rhs) {
        match comptime!(rhs.layout) {
            DummyLayout::RowMajor => fill_perpendicular(
                slice,
                rhs.handle.as_slice_mut(),
                Self::plane_unit(),
                Self::N,
                Self::K,
                Line::size(&slice[0]),
            ),
            DummyLayout::ColMajor => fill_parallel(
                slice,
                rhs.handle.as_slice_mut(),
                Self::plane_unit(),
                Self::K,
                Line::size(&slice[0]),
            ),
        }
    }

    fn init_output() -> Self::Out {
        let mut acc = Array::new(Self::N);
        for i in 0..Self::N {
            acc[i] = O::from_int(0);
        }
        OwnedTile::<O> {
            x: Self::M,
            y: Self::N,
            handle: acc,
            layout: as_dummy_layout(MatrixLayout::RowMajor),
        }
    }

    fn read_output<C: Numeric>(out: &Self::Out, slice: &mut SliceMut<'_, Line<C>>) {
        // TODO from config
        let line_size = Line::size(&slice[0]);
        let num_lines = Self::N / line_size;

        let row = Self::plane_unit();
        let unit_offset = row * num_lines;

        for col in 0..num_lines {
            let line = if comptime!(line_size == 1) {
                Line::cast_from(out.handle[col])
            } else {
                let mut line = Line::<C>::empty(line_size);
                for j in 0..line_size {
                    line[j] = C::cast_from(out.handle[col * line_size + j]);
                }
                line
            };

            slice[unit_offset + col] = line;
        }
    }
}

#[cube]
fn fill_parallel<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    row: u32,
    num_cols: u32,
    #[comptime] line_size: u32,
) {
    let num_lines = num_cols / line_size;

    for col in 0..num_lines {
        let line = slice_from[row * num_lines + col];
        #[unroll]
        for line_idx in 0..line_size {
            slice_to[col * line_size + line_idx] = line[line_idx];
        }
    }
}

#[cube]
fn fill_perpendicular<E: Numeric>(
    slice_from: &Slice<'_, Line<E>>,
    slice_to: &mut SliceMut<'_, E>,
    col: u32,
    num_cols: u32,
    num_rows: u32,
    #[comptime] line_size: u32,
) {
    let num_lines = num_cols / line_size;
    let col_idx = col / line_size;
    let line_idx = col % line_size;

    for row in 0..num_rows {
        let line = slice_from[row * num_lines + col_idx];
        let value = line[line_idx];
        slice_to[row] = value;
    }
}

impl<I: Numeric, O: Numeric, T: TmmConfig> Matmul<I, O> for PlaneMma32x32x32<I, O, T> {
    type Config = T;

    fn check_config(config: Self::Config) {
        assert!(config.plane_dim() == 32);
    }
}

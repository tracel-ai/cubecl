use std::cmp::max;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::{FragmentMask, FragmentMaskExpand};
use crate::components::tile::{RowVal, RowWise};

use crate::components::fragment::dummy_register::DummyRegisterAttentionMatmulConfig;
use crate::components::fragment::{AttentionMatmul, AttentionMatmulConfig as _};
use crate::components::fragment::{FragmentLayout, FragmentLayoutExpand};
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};

pub struct DummyRegisterAttentionMatmul;

#[derive(CubeType)]
/// Assumes:
/// - All elements of a unit are contiguous
/// - unit_size * plane_dim = total_size (not dim wise but in total count)
/// - There is never more than one row for one unit
pub struct ArrayTile<E: Numeric> {
    array: Array<E>,
    layout: ArrayTileLayout,
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum InnerLayout {
    /// Each unit has all its elements contiguous inside the same row
    ///
    ///  0,  0,  1,  1,  2,  2,  3,  3,
    ///  4,  4,  5,  5,  6,  6,  7,  7,
    ///  8,  8,  9,  9, 10, 10, 11, 11,
    /// 12, 12, 13, 13, 14, 14, 15, 15,
    /// 16, 16, 17, 17, 18, 18, 19, 19,
    /// 20, 20, 21, 21, 22, 22, 23, 23,
    /// 24, 24, 25, 25, 26, 26, 27, 27,
    /// 28, 28, 29, 29, 30, 30, 31, 31,
    Contiguous,
    /// Each unit spreads its elements along two rows
    ///
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    SplitRows,
}

#[cube]
impl<E: Numeric> ArrayTile<E> {
    pub fn new(layout: ArrayTileLayout) -> ArrayTile<E> {
        let array = Array::<E>::new(comptime!(layout.unit_size.0 * layout.unit_size.1));
        ArrayTile::<E> { array, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.unit_size.0 * self.layout.unit_size.1 {
            self.array[i] = E::from_int(0);
        }
    }
}

#[derive(CubeType, Copy, Clone)]
pub struct ArrayTileLayout {
    #[cube(comptime)]
    total_size: Coords2d,
    #[cube(comptime)]
    unit_size: Coords2d,
    #[cube(comptime)]
    num_units_per_row: u32,
    #[cube(comptime)]
    plane_dim: u32,
}

#[cube]
impl ArrayTileLayout {
    pub fn new(
        #[comptime] total_size: Coords2d,
        #[comptime] plane_dim: u32,
        #[comptime] inner_layout: InnerLayout,
    ) -> ArrayTileLayout {
        let total_elements = total_size.0 * total_size.1;
        let elements_per_unit = total_elements.div_ceil(plane_dim);

        let (num_rows_per_unit, num_cols_per_unit) = match inner_layout {
            InnerLayout::Contiguous => (1u32, elements_per_unit),
            InnerLayout::SplitRows => (2u32, elements_per_unit / 2u32),
        };
        let unit_size = (num_rows_per_unit, num_cols_per_unit);

        let num_units_per_row = comptime!(total_size.1 / unit_size.1);

        ArrayTileLayout {
            total_size,
            unit_size,
            num_units_per_row,
            plane_dim,
        }
    }
}

#[cube]
impl FragmentLayout for ArrayTileLayout {
    fn absolute_pos(&self, local_pos: Coords2d) -> Coords2d {
        let abs_row_index = {
            let row_0 = UNIT_POS_X / self.num_units_per_row;
            let row_jump = comptime!(self.plane_dim / self.num_units_per_row);

            local_pos.0 * row_jump + row_0
        };

        let abs_col_index = self.unit_size.1 * (UNIT_POS_X % self.num_units_per_row) + local_pos.1;

        (abs_row_index, abs_col_index)
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(self.total_size.1 / self.unit_size.1)
    }
}

#[cube]
impl<E: Float> FragmentOps<E> for ArrayTile<E> {
    type Layout = ArrayTileLayout;

    fn rowwise_max(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            let mut val = E::min_value();

            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                val = Max::max(val, self.array[index]);
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.unit_size.0,
            vals,
        }
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        let mut vals = Sequence::new();

        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            let mut val = E::from_int(0);

            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                val += self.array[index];
            }

            vals.push(RowVal::<E> { val });
        }

        RowWise::<E> {
            num_rows: self.layout.unit_size.0,
            vals,
        }
    }

    fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index] = self.array[index] * scale.index(r);
            }
        }
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        #[unroll]
        for r in 0..this.layout.unit_size.0 {
            let row_offset = r * this.layout.unit_size.1;
            #[unroll]
            for c in 0..this.layout.unit_size.1 {
                let index = row_offset + c;
                this.array[index] = this.array[index] * scale
                    + E::cast_from(mask.should_mask((r, c).runtime())) * E::min_value();
            }
        }
    }

    fn exp_diff(&mut self, val: &RowWise<E>) {
        #[unroll]
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            #[unroll]
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index] = Exp::exp(self.array[index] - val.index(r));
            }
        }
    }

    fn layout(&self) -> Self::Layout {
        self.layout
    }
}

#[cube]
impl<E: Numeric> FragmentMask for ArrayTile<E> {
    fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.array[local_pos.0 * self.layout.unit_size.1 + local_pos.1])
    }
}

#[cube]
fn array_tile_to_tmp_smem<E: Numeric>(
    array_tile: &ArrayTile<E>,
    #[comptime] num_planes: u32,
) -> SliceMut<E> {
    let tile_size = comptime!(array_tile.layout.total_size.0 * array_tile.layout.total_size.1);
    let mut tmp_smem = SharedMemory::<E>::new(comptime!(num_planes * tile_size));

    let start = UNIT_POS_Y * tile_size;
    let end = start + tile_size;
    let mut tmp_smem_slice = tmp_smem.slice_mut(start, end);

    if UNIT_POS_X == 0 {
        for i in 0..tile_size {
            tmp_smem_slice[i] = E::from_int(0);
        }
    }
    sync_cube();

    for r in 0..array_tile.layout.unit_size.0 {
        for c in 0..array_tile.layout.unit_size.1 {
            let (row, col) = array_tile.layout.absolute_pos((r, c));
            let index = row * array_tile.layout.total_size.1 + col;
            tmp_smem_slice[index] = array_tile.array[r * array_tile.layout.unit_size.1 + c];
        }
    }

    tmp_smem_slice
}

#[cube]
fn tmp_smem_to_array_tile<E: Numeric>(tmp_smem_slice: &SliceMut<E>, array_tile: &mut ArrayTile<E>) {
    for r in 0..array_tile.layout.unit_size.0 {
        for c in 0..array_tile.layout.unit_size.1 {
            let (row, col) = array_tile.layout.absolute_pos((r, c));
            let index = row * array_tile.layout.total_size.1 + col;
            array_tile.array[r * array_tile.layout.unit_size.1 + c] = tmp_smem_slice[index];
        }
    }
}

#[cube]
fn strided_tile_to_array_tile<E: Numeric, E2: Numeric>(
    strided_tile: &StridedTile<E>,
    array_tile: &mut ArrayTile<E2>,
) {
    for r in 0..array_tile.layout.unit_size.0 {
        for c in 0..array_tile.layout.unit_size.1 {
            let (row, col) = array_tile.layout.absolute_pos((r, c));
            array_tile.array[r * array_tile.layout.unit_size.1 + c] =
                E2::cast_from(strided_tile.get_line(row, col))
        }
    }
}

#[cube]
fn array_tile_to_slice<E: Numeric, E2: Numeric>(
    array_tile: &ArrayTile<E>,
    slice: &mut SliceMut<Line<E2>>,
) {
    for r in 0..array_tile.layout.unit_size.0 {
        for c in 0..array_tile.layout.unit_size.1 {
            let (row, col) = array_tile.layout.absolute_pos((r, c));
            let index = row * array_tile.layout.total_size.1 + col;
            slice[index] = Line::cast_from(array_tile.array[r * array_tile.layout.unit_size.1 + c]);
        }
    }
}

#[cube]
impl<AP: AttentionPrecision> AttentionMatmul<AP> for DummyRegisterAttentionMatmul {
    type Config = DummyRegisterAttentionMatmulConfig;

    type Query = ArrayTile<QT<AP>>;
    type KeyValue = ArrayTile<KVT<AP>>;
    type Mask = ArrayTile<MSK<AP>>;
    type Softmax = ArrayTile<SM<AP>>;
    type Accumulator = ArrayTile<ACC<AP>>;
    type FragmentLayout = ArrayTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> ArrayTileLayout {
        ArrayTileLayout::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().seq_kv,
            ),
            config.plane_dim(),
            config.inner_layout(),
        )
    }

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let tmp_lhs_smem_slice = array_tile_to_tmp_smem::<QT<AP>>(lhs, config.num_planes());
        let tmp_rhs_smem_slice = array_tile_to_tmp_smem::<KVT<AP>>(rhs, config.num_planes());
        let mut tmp_out_smem_slice = array_tile_to_tmp_smem::<SM<AP>>(out, config.num_planes());
        sync_cube();

        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_score_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = SM::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = tmp_lhs_smem_slice[i * k + ki];
                        let rhs_val = tmp_rhs_smem_slice[ki * n + j];
                        sum += SM::<AP>::cast_from(lhs_val) * SM::<AP>::cast_from(rhs_val);
                    }
                    tmp_out_smem_slice[i * n + j] = tmp_out_smem_slice[i * n + j] + sum;
                }
            }
        }

        sync_cube();
        tmp_smem_to_array_tile(&tmp_out_smem_slice, out);
        sync_cube();
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        sync_cube();
        let tmp_lhs_smem_slice = array_tile_to_tmp_smem::<SM<AP>>(lhs, config.num_planes());
        let tmp_rhs_smem_slice = array_tile_to_tmp_smem::<KVT<AP>>(rhs, config.num_planes());
        let mut tmp_out_smem_slice = array_tile_to_tmp_smem::<ACC<AP>>(out, config.num_planes());
        sync_cube();

        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_value_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = ACC::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = tmp_lhs_smem_slice[i * k + ki];
                        let rhs_val = tmp_rhs_smem_slice[ki * n + j];
                        sum += ACC::<AP>::cast_from(lhs_val) * ACC::<AP>::cast_from(rhs_val);
                    }
                    tmp_out_smem_slice[i * n + j] = tmp_out_smem_slice[i * n + j] + sum;
                }
            }
        }

        sync_cube();
        tmp_smem_to_array_tile(&tmp_out_smem_slice, out);
        sync_cube();
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(ArrayTileLayout::new(
            (
                comptime!(max(
                    config.attention_tile_size().head_dim,
                    config.attention_tile_size().seq_kv,
                )),
                comptime!(max(
                    config.attention_tile_size().seq_kv,
                    config.attention_tile_size().val_dim,
                )),
            ),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(ArrayTileLayout::new(
            (
                config.attention_tile_size().head_dim,
                config.attention_tile_size().seq_kv,
            ),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(ArrayTileLayout::new(
            (
                config.attention_tile_size().seq_kv,
                config.attention_tile_size().val_dim,
            ),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        ArrayTile::new(<Self as AttentionMatmul<AP>>::softmax_layout(config))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        ArrayTile::new(<Self as AttentionMatmul<AP>>::softmax_layout(config))
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        ArrayTile::new(ArrayTileLayout::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().val_dim,
            ),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let seq_q = config.attention_tile_size().seq_q;
        let head_dim = config.attention_tile_size().head_dim;

        ArrayTile::new(ArrayTileLayout::new(
            (seq_q, head_dim),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        strided_tile_to_array_tile(tile, fragment);

        sync_cube();
    }

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, rhs);

        sync_cube();
    }

    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, mask);

        sync_cube();
    }

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] _config: Self::Config) {
        softmax.zero();
        sync_cube();
    }

    fn zero_accumulator(acc: &mut Self::Accumulator) {
        acc.zero();
        sync_cube();
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] _config: Self::Config,
    ) {
        array_tile_to_slice(out, slice);
    }
}

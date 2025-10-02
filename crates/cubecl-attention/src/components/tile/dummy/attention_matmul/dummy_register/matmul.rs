use std::cmp::max;
use std::cmp::min;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;

use crate::components::tile::dummy::dummy_register::DummyRegisterAttentionMatmulConfig;
use crate::components::tile::dummy::{AttentionMatmul, AttentionMatmulConfig as _};
use crate::components::tile::{PlaneLayout, PlaneLayoutExpand};

pub struct DummyRegisterAttentionMatmul;

#[derive(CubeType)]
/// Assumes:
/// - All elements of a unit are contiguous
/// - unit_size * plane_dim = total_size (not dim wise but in total count)
/// - There is never more than one row for one unit
pub struct ArrayTile<E: Float> {
    array: Array<E>,
    #[cube(comptime)]
    total_size: Coords2d,
    #[cube(comptime)]
    unit_size: Coords2d,
}

#[cube]
impl<E: Float> ArrayTile<E> {
    pub fn new(#[comptime] total_size: Coords2d, #[comptime] plane_dim: u32) -> ArrayTile<E> {
        let total_elements = total_size.0 * total_size.1;
        let num_rows_per_unit = 1u32;
        let num_cols_per_unit = total_elements.div_ceil(plane_dim);
        let unit_size = (num_rows_per_unit, num_cols_per_unit);

        let array = Array::<E>::new(comptime!(unit_size.0 * unit_size.1));
        ArrayTile::<E> {
            array,
            total_size,
            unit_size,
        }
    }

    pub fn zero(&mut self) {
        for i in 0..self.unit_size.0 * self.unit_size.1 {
            self.array[i] = E::from_int(0);
        }
    }

    fn abs_row_index(&self) -> u32 {
        let num_units_per_col = self.total_size.1 / self.unit_size.1;
        UNIT_POS_X / num_units_per_col
    }

    fn abs_col_index(&self, c: u32) -> u32 {
        let num_units_per_col = self.total_size.1 / self.unit_size.1;
        self.unit_size.1 * (UNIT_POS_X % num_units_per_col) + c
    }
}

#[cube]
impl<E: Float> PlaneLayout for ArrayTile<E> {
    type E = E;

    fn num_local_rows(&self) -> comptime_type!(u32) {
        self.unit_size.0
    }

    fn num_local_cols(&self) -> comptime_type!(u32) {
        self.unit_size.1
    }

    fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(self.total_size.1 / self.unit_size.1)
    }

    fn get_at_coor(&self, r: u32, c: u32) -> E {
        self.array[r * self.unit_size.1 + c]
    }

    fn scale_at_coor(&mut self, r: u32, c: u32, factor: E) {
        let index = r * self.unit_size.1 + c;
        self.array[index] = self.array[index] * factor;
    }

    // fn scale_at_coor_tmp(&mut self, r: u32, c: u32, factor: E) {
    //     let index = r * self.unit_size.1 + c;
    //     // self.array[index] = E::from_int(0);

    //     self.array[index] = self.array[index] + E::new(0.001);
    // }

    fn exp_m_diff_at_coor(&mut self, r: u32, c: u32, val: E) {
        let index = r * self.unit_size.1 + c;
        self.array[index] = Exp::exp(self.array[index] - val);
    }
}

#[cube]
fn array_tile_to_tmp_smem<E: Float>(array_tile: &ArrayTile<E>, wqt: bool) -> SharedMemory<E> {
    let mut tmp_smem =
        SharedMemory::<E>::new(comptime!(array_tile.total_size.0 * array_tile.total_size.1));
    if UNIT_POS_X == 0 {
        for i in 0..comptime!(array_tile.total_size.0 * array_tile.total_size.1) {
            tmp_smem[i] = E::from_int(0);
        }
    }
    sync_cube();

    if !wqt {
        // assume unit_size.0 = 1
        for c in 0..array_tile.unit_size.1 {
            tmp_smem[array_tile.abs_row_index() * array_tile.total_size.1
                + array_tile.abs_col_index(c)] = array_tile.array[c];
        }
    }

    tmp_smem
}

#[cube]
fn change_smem<E: Float>(array_tile: &ArrayTile<E>, tmp_smem: &mut SharedMemory<E>) {
    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        tmp_smem
            [array_tile.abs_row_index() * array_tile.total_size.1 + array_tile.abs_col_index(c)] =
            E::from_int(999);
    }
    sync_cube();
}

#[cube]
fn tmp_smem_to_array_tile<E: Float>(tmp_smem: &SharedMemory<E>, array_tile: &mut ArrayTile<E>) {
    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        array_tile.array[c] = tmp_smem
            [array_tile.abs_row_index() * array_tile.total_size.1 + array_tile.abs_col_index(c)];
    }
}

#[cube]
fn strided_tile_to_array_tile<E: Float, E2: Float>(
    strided_tile: &StridedTile<E>,
    array_tile: &mut ArrayTile<E2>,
) {
    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        array_tile.array[c] = E2::cast_from(
            strided_tile.get_line(array_tile.abs_row_index(), array_tile.abs_col_index(c)),
        )
    }
}

#[cube]
fn array_tile_to_slice<E: Float, E2: Float>(
    array_tile: &ArrayTile<E>,
    slice: &mut SliceMut<Line<E2>>,
) {
    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        slice[array_tile.abs_row_index() * array_tile.total_size.1 + array_tile.abs_col_index(c)] =
            Line::cast_from(array_tile.array[c]);
    }
}

#[cube]
impl<AP: AttentionPrecision> AttentionMatmul<AP> for DummyRegisterAttentionMatmul {
    type Config = DummyRegisterAttentionMatmulConfig;

    type Query = ArrayTile<QT<AP>>;
    type KeyValue = ArrayTile<KVT<AP>>;
    type Softmax = ArrayTile<SM<AP>>;
    type Accumulator = ArrayTile<ACC<AP>>;

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        let mut tmp_lhs_smem = array_tile_to_tmp_smem::<QT<AP>>(lhs, false);
        let mut tmp_rhs_smem = array_tile_to_tmp_smem::<KVT<AP>>(rhs, false);
        let mut tmp_out_smem = array_tile_to_tmp_smem::<SM<AP>>(out, false);
        sync_cube();

        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_score_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = SM::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = tmp_lhs_smem[i * k + ki];
                        let rhs_val = tmp_rhs_smem[ki * n + j];
                        sum += SM::<AP>::cast_from(lhs_val) * SM::<AP>::cast_from(rhs_val);
                    }
                    tmp_out_smem[i * n + j] += sum;
                }
            }
        }

        sync_cube();
        tmp_smem_to_array_tile(&tmp_out_smem, out);
        sync_cube();
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        sync_cube();
        let mut tmp_lhs_smem = array_tile_to_tmp_smem::<SM<AP>>(lhs, false);
        let mut tmp_rhs_smem = array_tile_to_tmp_smem::<KVT<AP>>(rhs, false);
        let mut tmp_out_smem = array_tile_to_tmp_smem::<ACC<AP>>(out, true);
        sync_cube();

        // change_smem(lhs, &mut tmp_lhs_smem);
        // change_smem(rhs, &mut tmp_rhs_smem);

        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_value_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = ACC::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = tmp_lhs_smem[i * k + ki];
                        let rhs_val = tmp_rhs_smem[ki * n + j];
                        sum += ACC::<AP>::cast_from(lhs_val) * ACC::<AP>::cast_from(rhs_val);
                    }
                    tmp_out_smem[i * n + j] += sum;
                }
            }
        }

        sync_cube();
        tmp_smem_to_array_tile(&tmp_out_smem, out);
        sync_cube();
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(
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
        )
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(
            (
                config.attention_tile_size().head_dim,
                config.attention_tile_size().seq_kv,
            ),
            config.plane_dim(),
        )
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        ArrayTile::new(
            (
                config.attention_tile_size().seq_kv,
                config.attention_tile_size().val_dim,
            ),
            config.plane_dim(),
        )
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        ArrayTile::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().seq_kv,
            ),
            config.plane_dim(),
        )
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        ArrayTile::new(
            (
                config.attention_tile_size().seq_q,
                config.attention_tile_size().val_dim,
            ),
            config.plane_dim(),
        )
    }

    fn allocate_fill_query<EI: Float>(
        tile: &StridedTile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query {
        let seq_q = config.attention_tile_size().seq_q;
        let head_dim = config.attention_tile_size().head_dim;

        let mut query = ArrayTile::new((seq_q, head_dim), config.plane_dim());

        strided_tile_to_array_tile(tile, &mut query);

        sync_cube();
        query
    }

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, rhs);

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

        sync_cube();
    }
}

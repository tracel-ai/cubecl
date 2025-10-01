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
        for c in 0..self.unit_size.1 {
            self.array[c] = E::from_int(0);
        }
    }

    // pub fn get_at(&self, row: u32, col: u32) -> E {
    //     self.array[i]
    // }

    // pub fn assign(&mut self, i: u32, val: E) {
    //     self.array[i] = val;
    // }

    // pub fn add_assign(&mut self, i: u32, val: E) {
    //     self.array[i] += val;
    // }

    // pub fn mul_assign(&mut self, i: u32, val: E) {
    //     self.array[i] *= val;
    // }
}

#[cube]
impl<E: Float> PlaneLayout for ArrayTile<E> {
    type E = E;

    fn owned_rows_count(&self) -> comptime_type!(u32) {
        self.unit_size.0
    }

    fn is_owned(&self, row: u32) -> bool {
        UNIT_POS_X / self.unit_size.1 == row
    }

    fn total_rows_count(&self) -> comptime_type!(u32) {
        self.total_size.0
    }

    fn row_index(&self, r: u32) -> u32 {
        UNIT_POS_X / self.unit_size.1
    }

    fn num_cols(&self) -> comptime_type!(u32) {
        self.unit_size.1
    }

    fn col_index(&self, r: u32, c: u32) -> u32 {
        UNIT_POS_X % self.unit_size.1 + c
    }

    fn get_at_coor(&self, row: u32, col: u32) -> E {
        self.array[row * self.unit_size.1 + col]
    }

    fn scale_at_coor(&mut self, row: u32, col: u32, factor: E) {
        self.array[row * self.unit_size.1 + col] *= factor;
    }
}

#[cube]
fn array_tile_to_tmp_smem<E: Float>(array_tile: &ArrayTile<E>) -> SharedMemory<E> {
    let mut tmp_smem =
        SharedMemory::<E>::new(comptime!(array_tile.total_size.0 * array_tile.total_size.1));

    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        tmp_smem[array_tile.row_index(0u32) * array_tile.total_size.1
            + array_tile.col_index(0u32, c)] = array_tile.array[c];
    }

    tmp_smem
}

#[cube]
fn tmp_smem_to_array_tile<E: Float>(tmp_smem: &SharedMemory<E>, array_tile: &mut ArrayTile<E>) {
    // assume unit_size.0 = 1
    for c in 0..array_tile.unit_size.1 {
        array_tile.array[c] = tmp_smem
            [array_tile.row_index(0u32) * array_tile.total_size.1 + array_tile.col_index(0u32, c)];
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
            strided_tile.get_line(array_tile.row_index(0u32), array_tile.col_index(0u32, c)),
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
        slice[array_tile.row_index(0u32) * array_tile.total_size.1
            + array_tile.col_index(0u32, c)] = Line::cast_from(array_tile.array[c]);
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
        let tmp_lhs_smem = array_tile_to_tmp_smem::<QT<AP>>(lhs);
        let tmp_rhs_smem = array_tile_to_tmp_smem::<KVT<AP>>(rhs);
        let mut tmp_out_smem = array_tile_to_tmp_smem::<SM<AP>>(out);
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
        let tmp_lhs_smem = array_tile_to_tmp_smem::<SM<AP>>(lhs);
        let tmp_rhs_smem = array_tile_to_tmp_smem::<KVT<AP>>(rhs);
        let mut tmp_out_smem = array_tile_to_tmp_smem::<ACC<AP>>(out);
        sync_cube();

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

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        strided_tile_to_array_tile(tile, rhs);

        sync_cube();
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

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] config: Self::Config) {
        softmax.zero();
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

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        acc.zero();
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        array_tile_to_slice(out, slice);

        sync_cube();
    }

    // fn tmp_fill_accumulator(
    //     tile: &StridedTile<ACC<AP>>,
    //     acc: &mut Self::Accumulator,
    //     #[comptime] config: Self::Config,
    // ) {
    //     if UNIT_POS_X == 0 {
    //         let size = config.attention_tile_size().accumulator_size();
    //         for i in 0..size {
    //             acc.assign(i, tile.as_unlined(1u32).0[i]);
    //         }
    //     }

    //     sync_cube();
    // }

    // fn tmp_fill_prob(
    //     tile: &StridedTile<SM<AP>>,
    //     prob: &mut Self::Softmax,
    //     #[comptime] config: Self::Config,
    // ) {
    //     if UNIT_POS_X == 0 {
    //         let len = config.attention_tile_size().softmax_size();
    //         for i in 0..len {
    //             prob.assign(i, tile.as_unlined(1u32).0[i]);
    //         }
    //     }

    //     sync_cube();
    // }

    // fn tmp_write_softmax(
    //     softmax: &Self::Softmax,
    //     slice: &mut SliceMut<Line<SM<AP>>>,
    //     #[comptime] config: Self::Config,
    // ) {
    //     if UNIT_POS_X == 0 {
    //         let size = config.attention_tile_size().softmax_size();
    //         for i in 0..size {
    //             slice[i] = Line::cast_from(softmax.get_at(i));
    //         }
    //     }

    //     sync_cube();
    // }
}

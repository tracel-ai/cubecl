use std::cmp::max;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::dummy::dummy_register::DummyRegisterAttentionMatmulConfig;
use crate::components::tile::dummy::{AttentionMatmul, AttentionMatmulConfig as _};

/// Dummy AttentionMatmul implementation using simple arrays
/// Only lane 0 performs computations, other lanes idle
pub struct DummyRegisterAttentionMatmul;

#[cube]
impl<AP: AttentionPrecision> AttentionMatmul<AP> for DummyRegisterAttentionMatmul {
    type Config = DummyRegisterAttentionMatmulConfig;

    type Query = Array<QT<AP>>;
    type KeyValue = Array<KVT<AP>>;
    type Softmax = Array<SM<AP>>;
    type Accumulator = Array<ACC<AP>>;

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_score_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = SM::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = lhs[i * k + ki];
                        let rhs_val = rhs[ki * n + j];
                        sum += SM::<AP>::cast_from(lhs_val) * SM::<AP>::cast_from(rhs_val);
                    }
                    out[i * n + j] += sum;
                }
            }
        }

        sync_cube();
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = config.attention_tile_size().to_value_matmul_tile_size().into(); (m, n, k)};

            for i in 0..m {
                for j in 0..n {
                    let mut sum = ACC::<AP>::from_int(0);
                    for ki in 0..k {
                        let lhs_val = lhs[i * k + ki];
                        let rhs_val = rhs[ki * n + j];
                        sum += ACC::<AP>::cast_from(lhs_val) * ACC::<AP>::cast_from(rhs_val);
                    }
                    out[i * n + j] += sum;
                }
            }
        }

        sync_cube();
    }

    fn allocate_fill_query<EI: Numeric>(
        tile: &StridedTile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query {
        let seq_q = config.attention_tile_size().seq_q;
        let head_dim = config.attention_tile_size().head_dim;

        let mut query = Array::<QT<AP>>::new(seq_q * head_dim);

        if UNIT_POS_X == 0 {
            // Only lane 0 fills the data
            for q in 0..seq_q {
                for hd in 0..head_dim {
                    query[q * head_dim + hd] = QT::<AP>::cast_from(tile.get_line(q, hd));
                }
            }
        }

        sync_cube();
        query
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<KVT<AP>>::new(comptime!(max(
            config.attention_tile_size().key_size(),
            config.attention_tile_size().value_size(),
        )))
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<KVT<AP>>::new(config.attention_tile_size().key_size())
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        Array::<KVT<AP>>::new(config.attention_tile_size().value_size())
    }

    fn fill_key_value<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let size = config.attention_tile_size().key_size();
            for i in 0..size {
                rhs[i] = KVT::<AP>::cast_from(tile.as_unlined().0[i]);
            }
        }

        sync_cube();
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        Array::<SM<AP>>::new(config.attention_tile_size().softmax_size())
    }

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] config: Self::Config) {
        if UNIT_POS_X == 0 {
            let len = config.attention_tile_size().softmax_size();
            for i in 0..len {
                softmax[i] = SM::<AP>::from_int(0);
            }
        }
        sync_cube();
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        Array::<ACC<AP>>::new(config.attention_tile_size().accumulator_size())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        if UNIT_POS_X == 0 {
            let len = config.attention_tile_size().accumulator_size();
            for i in 0..len {
                acc[i] = ACC::<AP>::from_int(0);
            }
        }

        sync_cube();
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let size = config.attention_tile_size().accumulator_size();
            for i in 0..size {
                slice[i] = Line::cast_from(out[i]);
            }
        }

        sync_cube();
    }

    fn tmp_fill_accumulator(
        tile: &StridedTile<ACC<AP>>,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let size = config.attention_tile_size().accumulator_size();
            for i in 0..size {
                acc[i] = tile.as_unlined().0[i];
            }
        }

        sync_cube();
    }

    fn tmp_fill_prob(
        tile: &StridedTile<SM<AP>>,
        prob: &mut Self::Softmax,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let len = config.attention_tile_size().softmax_size();
            for i in 0..len {
                prob[i] = tile.as_unlined().0[i];
            }
        }

        sync_cube();
    }

    fn tmp_write_softmax(
        softmax: &Self::Softmax,
        slice: &mut SliceMut<Line<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let size = config.attention_tile_size().softmax_size();
            for i in 0..size {
                slice[i] = Line::cast_from(softmax[i]);
            }
        }

        sync_cube();
    }
}

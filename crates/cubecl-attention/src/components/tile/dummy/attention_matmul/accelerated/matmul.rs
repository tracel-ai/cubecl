use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::tile::dummy::accelerated::AcceleratedAttentionMatmulConfig;
use crate::components::tile::dummy::{AttentionMatmul, AttentionMatmulConfig as _};

/// Performs two matmuls with fragment reuse for key/value and score/prob
pub struct AcceleratedAttentionMatmul;

#[cube]
impl<AP: AttentionPrecision> AttentionMatmul<AP> for AcceleratedAttentionMatmul {
    type Config = AcceleratedAttentionMatmulConfig;
    type Query = cmma::Matrix<QT<AP>>;
    type KeyValue = cmma::Matrix<KVT<AP>>;
    type Softmax = cmma::Matrix<SM<AP>>;
    type Accumulator = cmma::Matrix<ACC<AP>>;

    fn score_matmul(
        lhs: &Self::Query,
        rhs: &Self::KeyValue,
        out: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<QT<AP>, KVT<AP>, SM<AP>, SM<AP>>(lhs, rhs, out, out);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        cmma::execute::<SM<AP>, KVT<AP>, ACC<AP>, ACC<AP>>(lhs, rhs, out, out);
    }

    fn allocate_fill_query<EI: Numeric>(
        tile: &StridedTile<EI>,
        #[comptime] config: Self::Config,
    ) -> Self::Query {
        let (slice, stride) = tile.as_unlined();
        let size = config.attention_tile_size().to_score_matmul_tile_size();

        if config.cast_query() {
            let query = unsafe {
                cmma::Matrix::<QT<AP>>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m(),
                    size.n(),
                    size.k(),
                    cmma::MatrixLayout::RowMajor,
                )
            };

            cmma::load(&query, &slice, stride);
            query
        } else {
            let tmp = unsafe {
                cmma::Matrix::<EI>::uninitialized(
                    cmma::MatrixIdent::A,
                    size.m(),
                    size.n(),
                    size.k(),
                    cmma::MatrixLayout::RowMajor,
                )
            };

            cmma::load(&tmp, &slice, stride);
            cmma::cast::<EI, QT<AP>>(&tmp)
        }
    }

    fn allocate_key(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<KVT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q,
                size.seq_kv,
                size.head_dim,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<KVT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                size.seq_q,
                size.val_dim,
                size.seq_kv,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_key_value(#[comptime] config: Self::Config) -> Self::KeyValue {
        let size = config.attention_tile_size();
        assert!(size.can_reuse_key_value());

        unsafe {
            cmma::Matrix::<KVT<AP>>::uninitialized(
                cmma::MatrixIdent::B,
                // m not relevant because it's a B
                size.seq_q,
                // n and k match key, but we are guaranteed that value takes the same space (albeit not the same shape)
                size.seq_kv,
                size.head_dim,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn fill_key_value<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(rhs, &slice, stride);
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<SM<AP>>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q,
                size.seq_kv,
                size.head_dim, // k, because we take accumulator point of view
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn zero_softmax(acc: &mut Self::Softmax, #[comptime] _config: Self::Config) {
        cmma::fill(acc, SM::<AP>::from_int(0));
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.attention_tile_size().to_value_matmul_tile_size();
        unsafe {
            cmma::Matrix::<ACC<AP>>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::Undefined,
            )
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        cmma::fill(acc, ACC::<AP>::from_int(0));
    }

    fn write_results<E: Numeric>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<ACC<AP>, E>(out);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }

    fn tmp_fill_accumulator(
        tile: &StridedTile<ACC<AP>>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load_with_layout(acc, &slice, stride, cmma::MatrixLayout::RowMajor);
    }
    fn tmp_fill_prob(
        tile: &StridedTile<SM<AP>>,
        prob: &mut Self::Softmax,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load_with_layout(prob, &slice, stride, cmma::MatrixLayout::RowMajor);
    }
    fn tmp_write_softmax(
        softmax: &Self::Softmax,
        slice: &mut SliceMut<Line<SM<AP>>>,
        #[comptime] config: Self::Config,
    ) {
        cmma::store(
            slice,
            softmax,
            config.attention_tile_size().seq_kv,
            cmma::MatrixLayout::RowMajor,
        );
    }
}

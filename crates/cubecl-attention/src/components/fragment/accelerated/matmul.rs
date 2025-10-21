use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;
use crate::components::fragment::accelerated::AcceleratedAttentionMatmulConfig;
use crate::components::fragment::{AttentionMatmul, AttentionMatmulConfig as _};
use crate::components::fragment::{FragmentLayout, FragmentLayoutExpand};
use crate::components::fragment::{FragmentMask, FragmentMaskExpand};
use crate::components::fragment::{FragmentOps, FragmentOpsExpand};
use crate::components::tile::RowWise;
use cubecl_std::tensor::layout::Coords2d;

/// Performs two matmuls with fragment reuse for key/value and score/prob
pub struct AcceleratedAttentionMatmul;

#[derive(CubeType)]
pub struct TODO;

#[cube]
impl FragmentLayout for TODO {
    fn absolute_pos(&self, _local_pos: Coords2d) -> Coords2d {
        todo!()
    }
    fn num_units_per_row(&self) -> comptime_type!(u32) {
        todo!()
    }
}

#[cube]
impl<E: Float> FragmentOps<E> for cmma::Matrix<E> {
    type Layout = TODO;

    fn rowwise_max(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_scale(&mut self, _val: &RowWise<E>) {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(_this: &mut Self, _scale: E, _mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, _val: &RowWise<E>) {
        todo!()
    }

    fn layout(&self) -> Self::Layout {
        todo!()
    }
}

#[cube]
impl<E: Numeric> FragmentMask for cmma::Matrix<E> {
    fn should_mask(&self, _local_pos: Coords2d) -> bool {
        todo!()
    }
}

#[cube]
impl<AP: AttentionPrecision> AttentionMatmul<AP> for AcceleratedAttentionMatmul {
    type Config = AcceleratedAttentionMatmulConfig;
    type Query = cmma::Matrix<QT<AP>>;
    type KeyValue = cmma::Matrix<KVT<AP>>;
    type Mask = cmma::Matrix<MSK<AP>>;
    type Softmax = cmma::Matrix<SM<AP>>;
    type Accumulator = cmma::Matrix<ACC<AP>>;
    type FragmentLayout = TODO;

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

    fn allocate_query(#[comptime] config: Self::Config) -> Self::Query {
        let size = config.attention_tile_size().to_score_matmul_tile_size();

        unsafe {
            cmma::Matrix::<QT<AP>>::uninitialized(
                cmma::MatrixIdent::A,
                size.m(),
                size.n(),
                size.k(),
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        let (slice, stride) = tile.as_unlined();

        cmma::load(fragment, &slice, stride);
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
                // n and k match key, and we assume value takes the same space
                size.seq_kv,
                size.head_dim,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<MSK<AP>>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q,
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

    fn fill_mask<E: Numeric>(
        _tile: &StridedTile<E>,
        _mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        let size = config.attention_tile_size();
        unsafe {
            cmma::Matrix::<SM<AP>>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                size.seq_q,
                size.seq_kv,
                size.head_dim, // k, because we take score matmul acc point of view
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

    fn zero_accumulator(acc: &mut Self::Accumulator) {
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

    fn softmax_layout(#[comptime] _config: Self::Config) -> Self::FragmentLayout {
        todo!()
    }
}

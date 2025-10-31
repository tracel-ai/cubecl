use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coords2d;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;

use crate::components::fragment::accelerated::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::fragment::accelerated::array_tile::ArrayTileLayout;
use crate::components::fragment::accelerated::array_tile::LocalTile;
use crate::components::fragment::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::fragment::{FragmentAttention, FragmentAttentionConfig as _};
use crate::components::fragment::{FragmentMask, FragmentMaskExpand};
use crate::components::fragment::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::RowWise;

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox
pub struct BlackboxAcceleratedFragmentAttention;

// TODO
// The hybrid fragments should not own their shared memory
// The shared memory should be decided at partition level
// ALSO, the partitions take too many registers, should not
// materialize all tiles at once

#[derive(CubeType)]
struct MaskTODO {}

#[cube]
impl FragmentMask for MaskTODO {
    type Layout = ArrayTileLayout;

    fn should_mask(&self, _local_pos: Coords2d) -> bool {
        false.runtime()
    }
}

#[derive(CubeType)]
/// Navigates between cmma fragment (for matmuls) and shared memory (for row wise ops)
pub struct HybridFragment<E: Float> {
    // For matmul
    fragment: cmma::Matrix<E>,
    // A slice because knows only the slot for this plane
    shared_memory: SliceMut<E>,
    // Where to perform operations in register
    local_tile: LocalTile<E>,
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for HybridFragment<E> {
    type Layout = ArrayTileLayout;

    fn layout(&self) -> Self::Layout {
        todo!()
    }

    fn rowwise_max(&self) -> RowWise<E> {
        todo!()
    }

    fn rowwise_sum(&self) -> RowWise<E> {
        todo!()
    }

    fn scale_and_mask<M: FragmentMask>(this: &mut Self, scale: E, mask: &M) {
        todo!()
    }

    fn exp_diff(&mut self, m: &RowWise<E>) {
        todo!()
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for HybridFragment<E> {
    fn rowwise_scale(&mut self, val: &crate::components::tile::RowWise<E>) {
        todo!()
    }
}

#[cube]
impl<AP: AttentionPrecision> FragmentAttention<AP> for BlackboxAcceleratedFragmentAttention {
    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    type Query = cmma::Matrix<QT<AP>>;
    type KeyValue = cmma::Matrix<KVT<AP>>;
    type Mask = MaskTODO;
    type Softmax = HybridFragment<SM<AP>>;
    type Accumulator = HybridFragment<ACC<AP>>;

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
        let out = &out.fragment;
        cmma::execute::<QT<AP>, KVT<AP>, SM<AP>, SM<AP>>(lhs, rhs, out, out);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let lhs = &lhs.fragment;
        let out = &out.fragment;
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

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        // let size = config.attention_tile_size();
        // unsafe {
        //     cmma::Matrix::<MSK<AP>>::uninitialized(
        //         cmma::MatrixIdent::Accumulator,
        //         size.seq_q,
        //         size.seq_kv,
        //         size.head_dim,
        //         cmma::MatrixLayout::RowMajor,
        //     )
        // }

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

    fn fill_query<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Query) {
        let (slice, stride) = tile.as_unlined();

        cmma::load(fragment, &slice, stride);
    }

    fn fill_key_value<E: Float>(
        tile: &StridedTile<E>,
        rhs: &mut Self::KeyValue,
        #[comptime] _config: Self::Config,
    ) {
        let (slice, stride) = tile.as_unlined();
        cmma::load(rhs, &slice, stride);
    }

    fn fill_mask<E: Numeric>(
        tile: &StridedTile<E>,
        mask: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        todo!()
    }

    fn zero_softmax(softmax: &mut Self::Softmax, #[comptime] _config: Self::Config) {
        cmma::fill(softmax, SM::<AP>::from_int(0));
    }

    fn zero_accumulator(acc: &mut Self::Accumulator) {
        cmma::fill(acc, ACC::<AP>::from_int(0));
    }

    fn write_results<E: Float>(
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
}

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::tile::StridedTile;

use crate::components::AttentionPrecision;
use crate::components::attention_types::*;

use crate::components::tile::accelerated::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated::hybrid_fragment::HybridFragment;
use crate::components::tile::accelerated::local_tile::LocalTile;
use crate::components::tile::accelerated::local_tile::LocalTileLayout;
use crate::components::tile::{FragmentAttention, FragmentAttentionConfig as _};

/// Uses accelerated instruction, but relies on shared memory for row-dependent computations
/// because the fragment layout is blackbox
pub struct BlackboxAcceleratedFragmentAttention;

#[cube]
impl<AP: AttentionPrecision> FragmentAttention<AP> for BlackboxAcceleratedFragmentAttention {
    type Config = BlackboxAcceleratedAttentionMatmulConfig;

    type Query = cmma::Matrix<QT<AP>>;
    type KeyValue = cmma::Matrix<KVT<AP>>;
    type Mask = LocalTile<SM<AP>>;
    type Softmax = HybridFragment<SM<AP>>;
    type SoftmaxRow = LocalTile<SM<AP>>;
    type Accumulator = HybridFragment<ACC<AP>>;

    type FragmentLayout = LocalTileLayout;

    fn softmax_layout(#[comptime] config: Self::Config) -> LocalTileLayout {
        LocalTileLayout::new(
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
        #[comptime] _config: Self::Config,
    ) {
        let out = &out.fragment;
        cmma::execute::<QT<AP>, KVT<AP>, SM<AP>, SM<AP>>(lhs, rhs, out, out);
    }

    fn value_matmul(
        lhs: &Self::Softmax,
        rhs: &Self::KeyValue,
        out: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
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
        let size = config.attention_tile_size();
        LocalTile::new(LocalTileLayout::new(
            (size.seq_q, size.seq_kv),
            config.plane_dim(),
            config.inner_layout(),
        ))
    }

    fn allocate_softmax(#[comptime] config: Self::Config) -> Self::Softmax {
        let size = config.attention_tile_size().to_score_matmul_tile_size();
        HybridFragment::new(size, config)
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let size = config.attention_tile_size().to_value_matmul_tile_size();
        HybridFragment::new(size, config)
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
        mask.fill_from_strided_tile(tile)
    }

    fn write_results<E: Float>(
        out: &Self::Accumulator,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<ACC<AP>, E>(&out.fragment);
        cmma::store(
            slice,
            &acc,
            config.attention_tile_size().val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}

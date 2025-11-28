use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::TileSize;

use crate::components::tile::accelerated::local_tile::{LocalTile, LocalTileLayout};
use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::components::tile::{RowWise, TileAttentionConfig as _};

#[derive(CubeType)]
/// Navigates between cmma fragment (for matmuls) and shared memory (for row wise ops)
pub struct HybridFragment<E: Float> {
    // For matmul
    pub fragment: cmma::Matrix<E>,
    // A slice because knows only the slot for this plane
    smem_slice: SliceMut<E>,
    // Where to perform operations in register
    local_tile: LocalTile<E>,
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<E: Float> HybridFragment<E> {
    pub fn new(
        #[comptime] tile_size: TileSize,
        #[comptime] config: BlackboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m,
                tile_size.n,
                tile_size.k,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (tile_size.m, tile_size.n),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = tile_size.m * tile_size.n;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let mut shared_memory = SharedMemory::new(config.num_planes() * smem_slot_size);
        let smem_slice =
            shared_memory.slice_mut(smem_slice_start, smem_slice_start + smem_slot_size);

        HybridFragment::<E> {
            fragment,
            smem_slice,
            local_tile,
            stride: tile_size.n,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.fragment, E::from_int(0));
    }
}

#[cube]
impl<E: Float> FragmentSoftmax<E> for HybridFragment<E> {
    type Layout = LocalTileLayout;
    type SoftmaxScore = cmma::Matrix<E>;
    type SoftmaxRowFormat = LocalTile<E>;
    type SoftmaxVal = cmma::Matrix<E>;

    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat {
        cmma::store(
            &mut self.smem_slice,
            &self.fragment,
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        self.local_tile.load_from_slice(&self.smem_slice.to_slice());

        sync_cube();

        &mut self.local_tile
    }

    fn update_from_rowwise(&mut self) {
        self.local_tile.store_to(&mut self.smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &self.fragment,
            &self.smem_slice.to_slice(),
            self.stride,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn zero(&mut self) {
        self.zero();
    }
}

#[cube]
impl<E: Float> FragmentAccumulator<E> for HybridFragment<E> {
    fn rowwise_scale(&mut self, val: &RowWise<E>) {
        let local_tile = self.rowwise_mut();
        local_tile.rowwise_scale(val);
        self.update_from_rowwise();
    }

    fn zero(&mut self) {
        self.zero();
    }
}

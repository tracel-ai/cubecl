use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{BatchOffsets, Coordinates, Dimensions, SharedMemories},
    compute_loop::compute_loop,
    config::CubeTiling2dConfig,
    load_shared_memory::load_to_shared_memories,
    tile::{loader::TileLoader, writer::TileWriter},
    write_output::write_to_output,
};

#[cube]
pub(crate) fn block_loop<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    out: &mut Tensor<Line<N>>,
    coordinates: Coordinates,
    offsets: BatchOffsets,
    shared: SharedMemories<N>,
    #[comptime] config: CubeTiling2dConfig,
    dims: Dimensions,
) {
    let mut results = init_results::<N>(config);
    let block_size_k = config.block_size_k;
    let n_loops = (dims.k + block_size_k - 1) / block_size_k;

    for k in 0..n_loops {
        let k = k * block_size_k;

        load_to_shared_memories::<N, TileLoader<N>>(
            lhs,
            rhs,
            coordinates,
            k,
            offsets,
            shared,
            config,
            dims,
        );

        sync_cube();

        compute_loop::<N>(coordinates, shared.lhs, shared.rhs, &mut results, config);

        sync_cube();
    }

    write_to_output::<N, TileWriter<N>>(out, &results, coordinates, offsets.out, dims, config);
}

#[cube]
fn init_results<N: Numeric>(#[comptime] config: CubeTiling2dConfig) -> Array<N> {
    let tile_size = config.tile_size;
    let unroll = config.unroll_tile;

    let mut results = Array::<N>::new(tile_size * tile_size);
    #[unroll(unroll)]
    for i in 0..tile_size * tile_size {
        results[i] = N::from_int(0);
    }

    results
}

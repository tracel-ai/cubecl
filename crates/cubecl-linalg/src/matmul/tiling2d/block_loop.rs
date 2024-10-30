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
pub(crate) fn block_loop<F: Float>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    out: &mut Tensor<Line<F>>,
    coordinates: Coordinates,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    #[comptime] config: CubeTiling2dConfig,
    dims: Dimensions,
) {
    let mut results = init_results::<F>(config);
    let block_size_k = config.block_size_k;
    let n_loops = (dims.k + block_size_k - 1) / block_size_k;

    for k in 0..n_loops {
        let k = k * block_size_k;

        load_to_shared_memories::<F, TileLoader<F>>(
            lhs,
            rhs,
            coordinates,
            k,
            offsets,
            shared,
            config,
            dims,
        );

        sync_units();

        compute_loop::<F>(coordinates, shared.lhs, shared.rhs, &mut results, config);

        sync_units();
    }

    write_to_output::<F, TileWriter<F>>(out, &results, coordinates, offsets.out, dims, config);
}

#[cube]
fn init_results<F: Float>(#[comptime] config: CubeTiling2dConfig) -> Array<F> {
    let tile_size = config.tile_size;
    let unroll = config.unroll_tile;

    let mut results = Array::<F>::new(tile_size * tile_size);
    #[unroll(unroll)]
    for i in 0..tile_size * tile_size {
        results[i] = F::new(0.);
    }

    results
}

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
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    coordinates: Coordinates,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    dims: Dimensions,
) {
    let mut results = init_results::<F>(config);
    let block_size_k = Comptime::runtime(Comptime::map(config, |c| c.block_size_k));
    let n_loops = (dims.k + block_size_k - 1) / block_size_k;

    for k in range(0u32, n_loops, Comptime::new(false)) {
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
fn init_results<F: Float>(config: Comptime<CubeTiling2dConfig>) -> Array<F> {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let unroll = Comptime::map(config, |c| c.unroll_tile);

    let mut results = Array::<F>::new(Comptime::get(tile_size * tile_size));
    for i in range(0u32, Comptime::get(tile_size * tile_size), unroll) {
        results[i] = F::new(0.);
    }

    results
}

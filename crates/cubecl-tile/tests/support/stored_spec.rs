use cubecl_tile::layout::StoragePartitioning;
use cubecl_tile::{Dimension, Geometry, Projection, TileSpec};

/// The spec of a buffer laid out as `geometry` in `storage`'s tiles, its axes the pieces of
/// `dimensions`.
pub fn stored_tiles_spec(
    dimensions: &[Dimension],
    storage: &StoragePartitioning,
    geometry: &Geometry,
) -> TileSpec {
    let pieces = geometry
        .pieces(dimensions)
        .expect("the stored tiles should name every dimension");
    let mut spec = TileSpec::new(Projection::tiled(dimensions, &pieces));
    spec.stored_tiles = storage.tiles().to_vec();
    spec
}

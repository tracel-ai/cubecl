//! Unit tests for [`Launcher`]: geometry read off the concrete space, kernel space dynamic.

use super::{Form, implied};
use cubecl::{prelude::*, zspace::Tiling};
use cubecl_tile::launch::{BoundaryPolicy, Refusal};
use cubecl_tile::{
    Axis, Geometry, Level, Partitioning, Projection, Space, TileSpec,
    kind::{Boundary, Storage},
    layout::{Divisor, Offset, PhysicalAxisMap, Scale, StorageLevels, VectorTile},
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

#[test]
fn launcher_geometry_matches_concrete_space() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // X: 64/16 cube tiles, Y: 64/32, nothing on Z.
    match launch.cube_count() {
        CubeCount::Static(x, y, z) => assert_eq!((x, y, z), (4, 2, 1)),
        _ => panic!("launcher geometry should be static"),
    }
    // Planes: within a 16×32 cube tile, 2×1 leaves of 8×32, each as wide as the device's plane.
    let plane = client.properties().hardware.plane_size_max;
    assert_eq!(launch.cube_dim(), CubeDim::new_2d(plane, 2));
}

#[test]
fn launcher_kernel_space_is_dynamic_concrete_is_not() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    for axis in [M, N, K] {
        assert!(launch.partitioning().space().is_dynamic(axis));
        assert!(!launch.space().is_dynamic(axis));
    }
    assert_eq!(launch.space().extent(M), 64);
}

#[test]
fn dynamic_along_frees_only_the_listed_axes() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[M, K]),
        )
    };

    assert!(launch.partitioning().space().is_dynamic(M));
    assert!(launch.partitioning().space().is_dynamic(K));
    assert!(!launch.partitioning().space().is_dynamic(N));
}

/// An axis the space does not have would be dropped silently, leaving the kernel specialized along
/// the axis the caller meant to free.
#[test]
#[should_panic(expected = "is not an axis of this space")]
fn dynamic_along_unknown_axis_panics() {
    let client = cubecl::test_device().client();
    let (space, levels) = batched_space(1, 1, 64, 64, 16);
    let _ = implied(
        &client,
        Partitioning::new(space, levels),
        Form::DynamicAlong(&[Axis(9)]),
    );
}

/// The footgun the launcher removes: geometry read after `all_dynamic` has no extents to
/// read. Pinned so the two-step order stays a real constraint, not a stale comment.
#[test]
#[should_panic(expected = "Dynamic")]
fn geometry_after_dynamic_panics() {
    let (space, levels) = batched_space(1, 1, 64, 64, 16);
    let _ = implied(
        &cubecl::test_device().client(),
        Partitioning::new(space.all_dynamic(), levels),
        Form::Static,
    )
    .cube_count();
}

// ---- Launcher::arg ---------------------------------------------------------

const B0: Axis = Axis(3);
const B1: Axis = Axis(4);

fn binding(client: &Client, shape: &[usize]) -> TensorBinding {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let len: usize = shape.iter().product();
    TensorBinding {
        handle: client.empty(len * size_of::<f32>()).binding(),
        strides: strides.into(),
        shape: shape.to_vec().into(),
        tiling: Tiling::UNTILED,
    }
}

/// A CPU-matmul-shaped scheme: two batch axes riding one-per-cube on Z, 16×32 cube tiles on
/// X/Y, 8×32 plane leaves with `leaf_k = 4`. Two planes a cube, so the cube fits a device
/// whose cube holds as few units as a small CPU runner's cores.
fn batched_space(b0: usize, b1: usize, m: usize, n: usize, k: usize) -> (Space, Vec<Level>) {
    (
        Space::new(&[(B0, b0), (B1, b1), (M, m), (N, n), (K, k)]),
        cubecl_tile::Levels::leaf(&[(M, 8), (N, 32), (K, 4)])
            .walk_every(&[K])
            .planes(&[(M, 2)])
            .cubes(&[M, N])
            .batches(&[B0, B1])
            .build(),
    )
}

#[test]
fn arg_derives_check_from_subspace_overhang() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4); M and N divide everywhere.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    let touches_k = launch
        .arg(binding(&client, &[64, 18]), f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert!(touches_k.spec.is_checked());
    // Only the axis that overhangs carries the mask: M divides, so its coordinate is settled and
    // masking it would cost a comparison on every access that can never fail.
    assert_eq!(
        touches_k.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Zero)]
    );

    let avoids_k = launch
        .arg(binding(&client, &[64, 64]), f32::elem_type_native())
        .axes(&[M, N])
        .build()
        .unwrap();
    assert!(!avoids_k.spec.is_checked());
    assert!(avoids_k.spec.boundaries.is_empty());

    // An explicit override still wins over the derivation.
    let forced = launch
        .arg(binding(&client, &[64, 18]), f32::elem_type_native())
        .axes(&[M, K])
        .boundary(BoundaryPolicy::Unchecked)
        .build()
        .unwrap();
    assert!(!forced.spec.is_checked());
    assert!(forced.spec.boundaries.is_empty());
}

/// A storage-tiled operand is addressed by fewer coordinates than its buffer has dims, and the
/// boundary list follows the coordinates: sized off the physical rank it would land its per-axis
/// modes on the grid fragments, which no [`Window`] ever reads.
#[test]
fn arg_sizes_boundaries_by_coordinate_rank_under_storage_tiling() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4), so K's coordinate is the one that needs the mode.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // 2 coordinate axes (M, K), tiled into 4 physical buffer dims: 8*8 = 64 and 5*4 = 20, the
    // buffer padded to whole blocks of the leaf's own (8, 4).
    let mut tiled = binding(&client, &[8, 5, 8, 4]);
    tiled.tiling = Tiling::new(&[2, 2]).unwrap();
    let tiled = launch
        .arg(tiled, f32::elem_type_native())
        .axes(&[M, K])
        .boundary(BoundaryPolicy::Every(Boundary::Clamp))
        .build()
        .unwrap();

    assert_eq!(tiled.spec.projection.physical_rank(), 4);
    assert_eq!(tiled.spec.projection.coordinate_rank(), 2);
    assert_eq!(
        tiled.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Clamp)]
    );
}

/// The tiling is a fact of the tensor, read off its binding: no launch has to be told how its
/// own inputs are stored, and the projection it derives is the tiled one.
#[test]
fn arg_reads_the_storage_tiling_off_the_binding() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 8);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // The buffer says for itself that both its dims are stored two fragments deep.
    let mut tiled = binding(&client, &[8, 2, 8, 4]);
    tiled.tiling = Tiling::new(&[2, 2]).unwrap();
    let off_the_tensor = launch
        .arg(tiled, f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();

    assert_eq!(
        off_the_tensor.spec.projection,
        Projection::tiled(&[M, K], &[M, K, M, K])
    );
}

/// A storage tile is the tile of one of the kernel's levels: the space owns the block size,
/// so the launch names that level and the kernel descends into the block rather than checking
/// a boundary it cannot see. An untiled operand is one block over the whole buffer.
#[test]
fn arg_matches_a_storage_tile_to_the_level_it_is() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 8);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // Storage of (8, 4) are the leaf, the third level's tile.
    let mut leaf_tiles = binding(&client, &[8, 2, 8, 4]);
    leaf_tiles.tiling = Tiling::new(&[2, 2]).unwrap();
    let leaf = launch
        .arg(leaf_tiles, f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert_eq!(leaf.spec.storage, Storage::Tiled(Some(2)));

    // Storage of (16, K whole) are the cube's tile, the first level's, where that level hands K
    // down static: a level that leaves an axis whole matches a tile stored whole along it. With
    // K dynamic in the kernel the level has no fixed tile, so nothing matches.
    let cube_tiles = || {
        let mut tiles = binding(&client, &[4, 8, 16]);
        tiles.tiling = Tiling::new(&[2, 1]).unwrap();
        tiles
    };
    let fixed = {
        let (space, levels) = batched_space(1, 1, 64, 64, 8);
        implied(&client, Partitioning::new(space, levels), Form::Static)
    };
    let cube = fixed
        .arg(cube_tiles(), f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert_eq!(cube.spec.storage, Storage::Tiled(Some(0)));
    let cube = launch
        .arg(cube_tiles(), f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert_eq!(cube.spec.storage, Storage::Tiled(None));

    let plain = launch
        .arg(binding(&client, &[64, 8]), f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert_eq!(plain.spec.storage, Storage::Strided);
}

/// A `[m, k]` buffer of 64 by 64 stored in NVFP4-shaped tiles: a word of 8 along `K`, then 2
/// words along `K` by 2 rows, then 4 of those along `K` by 4 rows, the tiles down `K`.
fn nvfp4_shaped(client: &Client) -> TensorBinding {
    let geometry = StorageLevels::new(&[(K, 8)])
        .tile(&[(K, 2), (M, 2)])
        .tile(&[(K, 4), (M, 4)])
        .grid(&[K, M])
        .physical(&[(M, 64), (K, 64)])
        .unwrap();
    let mut tiled = binding(client, geometry.shape());
    tiled.strides = geometry.strides().into();
    tiled.tiling = geometry.tiling();
    tiled
}

/// A buffer stored in tiles says so in its spec, and a load of 32 values binds as the 16 by 2
/// rectangle those tiles make: how the buffer is stored is the operand's to state, what a load
/// covers the kernel's to pick out of it.
#[test]
fn arg_binds_a_load_across_the_tiles_a_buffer_stores() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 64);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let arg = launch
        .arg(nvfp4_shaped(&client), f32::elem_type_native())
        .axes(&[M, K])
        .vectorize(32)
        .build()
        .unwrap();
    assert_eq!(
        arg.spec.stored_tiles,
        vec![(K, 8), (K, 2), (M, 2), (K, 4), (M, 4)]
    );
    assert_eq!(
        VectorTile::new(&arg.spec.stored_tiles, K, 32)
            .unwrap()
            .extents(),
        &[(K, 16), (M, 2)]
    );
}

/// A load that would cut a stored tile does not bind: four values are half a word.
#[test]
fn arg_refuses_a_load_that_cuts_a_stored_tile() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 64);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let refused = launch
        .arg(nvfp4_shaped(&client), f32::elem_type_native())
        .axes(&[M, K])
        .vectorize(4)
        .build()
        .err();
    assert!(
        matches!(refused, Some(Refusal::WidthNotServed { width: 4, .. })),
        "{refused:?}"
    );
}

/// A tensor whose block is no level's tile is still an operand: (16, 4) is the cube's M with the
/// leaf's K, which no level cuts to, so no window is known to lie inside one storage tile and every
/// read goes through the layout walk.
#[test]
fn arg_reads_a_storage_tile_that_is_no_level_through_its_layout() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 8);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let mut tiled = binding(&client, &[4, 2, 16, 4]);
    tiled.tiling = Tiling::new(&[2, 2]).unwrap();
    let arg = launch
        .arg(tiled, f32::elem_type_native())
        .axes(&[M, K])
        .build()
        .unwrap();
    assert_eq!(arg.spec.storage, Storage::Tiled(None));
}

/// A batch dim ahead of the tiled block: the metadata describes every logical dim, the operand's
/// subspace only the inner two, and the leading one carries through as a plain batch dim that
/// the block is not matched on.
#[test]
fn arg_reads_a_tiling_stated_over_batch_dims_too() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(3, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    let mut tiled = binding(&client, &[3, 8, 5, 8, 4]);
    tiled.tiling = Tiling::new(&[1, 2, 2]).unwrap();
    let arg = launch
        .arg(tiled, f32::elem_type_native())
        .axes(&[M, K])
        .batches(&[B0, B1])
        .build()
        .unwrap();

    assert_eq!(arg.spec.projection.physical_rank(), 5);
    assert_eq!(arg.spec.projection.coordinate_rank(), 3);
    assert_eq!(arg.spec.storage, Storage::Tiled(Some(2)));
}

#[test]
fn arg_right_aligns_batches_and_drops_size_one() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(4, 3, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // One leading dim: right-aligns to B1 (the trailing axis of the full list).
    let one_batch = launch
        .arg(binding(&client, &[3, 64, 16]), f32::elem_type_native())
        .axes(&[M, K])
        .batches(&[B0, B1])
        .build()
        .unwrap();
    assert!(one_batch.spec.axes().contains(&B1));
    assert!(!one_batch.spec.axes().contains(&B0));

    // A size-1 dim drops out entirely (broadcast omission).
    let broadcast = launch
        .arg(binding(&client, &[1, 64, 16]), f32::elem_type_native())
        .axes(&[M, K])
        .batches(&[B0, B1])
        .build()
        .unwrap();
    assert!(!broadcast.spec.axes().contains(&B0));
    assert!(!broadcast.spec.axes().contains(&B1));
}

// ---- Launcher::geometry ----------------------------------------------------

/// The pair the API exists for: a destination with no tensor to bind derives *the same* spec a
/// bound one does. If these ever diverge, a fused store walks a different tile than the unfused
/// kernel it replaces, which is the whole thing `build_spec` is a shared derivation to prevent.
#[test]
fn spec_derives_what_a_bound_operand_derives() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4), so the derivation has a check to arm and something to say.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let geometry = Geometry::new(&[(64, 18), (18, 1)]);

    let bound = launch
        .arg(binding(&client, geometry.shape()), f32::elem_type_native())
        .axes(&[M, K])
        .vectorize(1)
        .build()
        .unwrap();
    let derived = launch
        .unbound(&geometry)
        .axes(&[M, K])
        .vectorize(1)
        .build_spec()
        .unwrap();

    // The spec alone: `vector_size` is the argument echoed back by both, so comparing the two
    // would compare `1` with `1` and pass however far apart the derivations drifted.
    assert_eq!(derived.spec, bound.spec);
}

/// A bound operand tunes the same knobs an unbound one does, so what comes back is the builder,
/// not a finished spec. A fused operand that knows its bounds better than the concrete overhang
/// states it *on* the derivation; a spec hand-built beside it is the drift `build_spec` removes.
#[test]
fn spec_tunes_what_a_bound_operand_tunes() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4), so the derivation arms a check there is something to disarm.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let geometry = Geometry::new(&[(64, 18), (18, 1)]);

    let derived = launch
        .unbound(&geometry)
        .axes(&[M, K])
        .vectorize(1)
        .build_spec()
        .unwrap();
    assert!(derived.spec.is_checked());

    let tuned = launch
        .unbound(&geometry)
        .axes(&[M, K])
        .vectorize(1)
        .boundary(BoundaryPolicy::Unchecked)
        .build_spec()
        .unwrap();
    assert!(!tuned.spec.is_checked());
}

/// The geometry comes back with the spec, because that is the pair [`Tile::of_sink`] takes and
/// re-deriving it at the call site is the drift `build_spec` removes. No batch axes are stated, so
/// nothing is dropped, yet the caller reads the settled pair rather than trusting the stated one.
///
/// See [`spec_settles_a_broadcast_batch_dim_away`] for the case where the two differ.
#[test]
fn spec_returns_the_geometry_it_settled_on() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    let derived = launch
        .unbound(&Geometry::new(&[(64, 16), (16, 1)]))
        .axes(&[M, K])
        .vectorize(1)
        .build_spec()
        .unwrap();

    assert_eq!(derived.geometry.shape(), [64, 16]);
    assert_eq!(derived.geometry.strides(), [16, 1]);
    assert_eq!(
        derived.spec.projection.physical_rank(),
        derived.geometry.rank()
    );
}

/// A leading broadcast dim is refused, not silently dropped. `geometry` labels the operand's
/// own axes and no batches are stated, so a dim past them has no axis to belong to and the
/// derivation that *would* drop it never runs; the rank fails at launch, not as a misplaced store.
#[test]
fn spec_refuses_a_dim_it_cannot_label() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    let refused = launch
        .unbound(&Geometry::new(&[(1, 1024), (64, 16), (16, 1)]))
        .axes(&[M, K])
        .vectorize(1)
        .build_spec()
        .err();
    assert!(
        matches!(refused, Some(Refusal::UnlabelledBatchDims { axes: 0, .. })),
        "{refused:?}"
    );
}

/// A broadcast batch dim is dropped from the geometry that comes back, which is the case the
/// stated and the settled pair actually part company over.
///
/// [`batches`](cubecl_tile::StridedTileSource::batches) right-aligns above the operand's own axes
/// as for a bound operand, and a size-one batch dim drops out. A caller handing [`Tile::of_sink`]
/// the *stated* geometry would address a rank the projection lost; the settled one is returned.
#[test]
fn spec_settles_a_broadcast_batch_dim_away() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(4, 3, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // Three dims stated, the leading one broadcast over B1.
    let derived = launch
        .unbound(&Geometry::new(&[(1, 1024), (64, 16), (16, 1)]))
        .axes(&[M, K])
        .batches(&[B0, B1])
        .vectorize(1)
        .build_spec()
        .unwrap();

    assert!(!derived.spec.axes().contains(&B1));
    assert_eq!(derived.geometry.shape(), [64, 16]);
    assert_eq!(derived.geometry.strides(), [16, 1]);
    assert_eq!(
        derived.spec.projection.physical_rank(),
        derived.geometry.rank()
    );
}

/// A width the operand cannot be served in is refused where it is stated, not left to truncate.
///
/// The kernel re-expresses the geometry in lines, a coarser stride becoming `stride / v`. A `v`
/// that does not divide it addresses a fraction of the operand: in bounds, no fault, wrong numbers.
/// `Launcher::vector_size` derives a dividing width for a bound operand; a stated one has none.
#[test]
fn spec_refuses_a_width_the_geometry_cannot_serve() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    // Row stride 17: an odd number of scalars, so no whole number of 2-wide lines steps a row.
    let refused = launch
        .unbound(&Geometry::new(&[(64, 17), (16, 1)]))
        .axes(&[M, K])
        .vectorize(2)
        .build_spec()
        .err();
    assert!(
        matches!(refused, Some(Refusal::WidthNotServed { width: 2, .. })),
        "{refused:?}"
    );
}

// ---- StridedTileSource::gathered -------------------------------------------

/// A convolution-shaped input over `batched_space`: output positions `M` at `stride` and taps `K`
/// at `dilation` share one physical dim, channels `N` ride the other. Three logical axes over a
/// rank-2 buffer, which no list of dim labels describes.
fn window(stride: usize, dilation: usize, offset: impl Into<Offset>) -> Projection {
    Projection::new(
        &[M, K, N],
        &[
            PhysicalAxisMap::affine(&[(M, stride), (K, dilation)]).shifted(offset),
            PhysicalAxisMap::of(N),
        ],
    )
}

/// The mapping reaches the spec as given: the buffer keeps its two dims and the tile spans the
/// three logical axes, in the projection's own order.
#[test]
fn arg_gathered_states_its_own_mapping() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };

    let input = launch
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .build()
        .unwrap();

    assert_eq!(input.spec.axes(), &[M, K, N]);
    assert_eq!(input.spec.projection, window(1, 1, 0));
    assert_eq!(input.spec.projection.physical_rank(), 2);
    // Nothing overhangs and the origin cannot go negative, so the gather stays unchecked.
    assert!(input.spec.boundaries.is_empty());
}

/// An overhanging axis arms the check through the affine map just as it does through a label: the
/// tap axis `K` is one of the two the gathered dim is addressed by.
#[test]
fn arg_gathered_derives_check_from_overhang() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4).
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };

    let input = launch
        .arg(binding(&client, &[81, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .build()
        .unwrap();
    // The gathered coordinate takes the mask; N is its own coordinate and divides, so it is
    // settled whatever the gather does.
    assert_eq!(
        input.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    // An explicit override still wins over the derivation.
    let forced = launch
        .arg(binding(&client, &[81, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .boundary(BoundaryPolicy::Unchecked)
        .build()
        .unwrap();
    assert!(forced.spec.boundaries.is_empty());
}

/// A padded window reads before the buffer's start whatever the tiling divides, so the derivation
/// arms on the offset's sign alone. A forward shift cannot underflow and stays unchecked.
#[test]
fn arg_gathered_derives_check_from_underflow() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };

    let padded = launch
        .arg(binding(&client, &[64, 64]), f32::elem_type_native())
        .gathered(window(1, 1, -1))
        .build()
        .unwrap();
    assert_eq!(
        padded.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    // A runtime offset's sign is unknown at launch, so it arms the guard conservatively.
    let dynamic = launch
        .arg(binding(&client, &[64, 64]), f32::elem_type_native())
        .gathered(window(1, 1, Offset::Dynamic))
        .build()
        .unwrap();
    assert_eq!(
        dynamic.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    let shifted = launch
        .arg(binding(&client, &[96, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 1))
        .build()
        .unwrap();
    assert!(shifted.spec.boundaries.is_empty());
}

/// One map per buffer dim: a mapping that addresses fewer dims than the operand has would read
/// every coarser stride as if it were the operand's own.
#[test]
fn arg_gathered_rank_mismatch_is_refused() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };
    let refused = launch
        .arg(binding(&client, &[4, 79, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .build()
        .err();
    assert!(
        matches!(
            refused,
            Some(Refusal::GatherRankMismatch { mapped: 2, rank: 3 })
        ),
        "{refused:?}"
    );
}

/// The gather contract runs on the caller's thread now that the builder knows the served width:
/// the innermost dim is addressed in lines, so it must be one logical axis at coefficient 1.
#[test]
fn arg_gathered_validates_the_innermost_dim() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[M]),
        )
    };
    let refused = launch
        .arg(binding(&client, &[64, 79]), f32::elem_type_native())
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::affine(&[(N, 1), (K, 2)]),
            ],
        ))
        .vectorize(4)
        .boundary(BoundaryPolicy::Unchecked)
        .build()
        .err();
    assert!(
        matches!(refused, Some(Refusal::GatherInnermostNotInLines)),
        "{refused:?}"
    );
}

/// An axis sharing its dim with another has no extent of its own to read back here, but the
/// operand may ride it [`Dynamic`] anyway: whoever maps it identically states its size when the op
/// walks it. The builder sees one operand; `witnessed_space` flags an unanswered axis at expansion.
#[test]
fn arg_gathered_dynamic_axis_is_accepted() {
    let client = cubecl::test_device().client();
    // `K` shares the gathered dim with `M`, so neither reads an extent off *this* operand.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N, K]),
        )
    };
    let _ = launch
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .build()
        .unwrap();
}

/// The axis a gather does identity-map still reads its own extent, so it is free to stay runtime
/// alongside the two that share a dim.
#[test]
fn arg_gathered_identity_axis_may_stay_dynamic() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };

    let input = launch
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .build()
        .unwrap();
    assert_eq!(input.spec.axes(), &[M, K, N]);
    assert!(launch.partitioning().space().is_dynamic(N));
    assert!(!launch.partitioning().space().is_dynamic(M));
}

/// A runtime coefficient sizes its compacted window by its declared `max`, so the smem it stages
/// into holds every window the launch can then ask for.
#[test]
fn arg_gathered_dynamic_coefficient_stages_to_its_bound() {
    let client = cubecl::test_device().client();
    let staged = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            cubecl_tile::Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::DynamicAlong(&[N]),
    );
    let _ = staged
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::scaled(&[
                    (M, Scale::Dynamic { max: 2 }),
                    (K, Scale::Dynamic { max: 3 }),
                ]),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build()
        .unwrap();
}

/// A static rational projection stages uncompacted into shared memory.
#[test]
fn arg_gathered_rational_stages() {
    let client = cubecl::test_device().client();
    let staged = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            cubecl_tile::Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::DynamicAlong(&[N]),
    );
    let _ = staged
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::affine(&[(M, 3), (K, 4)]).over(4),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build()
        .unwrap();
}

/// A dynamic divisor stages against its `min`, the smallest divisor and so the widest window any
/// launch can ask for.
#[test]
fn arg_gathered_dynamic_divisor_stages_to_its_bound() {
    let client = cubecl::test_device().client();
    let staged = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            cubecl_tile::Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::DynamicAlong(&[N]),
    );
    let _ = staged
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::affine(&[(M, 3), (K, 4)]).over(Divisor::Dynamic { min: 4 }),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build()
        .unwrap();
}

/// The same shape with a divisor its coefficients cancel: `⌊(8m + 4k)/4⌋` steps like `2m + k`, so
/// `over` reduces it away and what reaches the stage is a plain strided gather, which compacts.
#[test]
fn arg_gathered_cancelling_divisor_stages() {
    let client = cubecl::test_device().client();
    let staged = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            cubecl_tile::Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::DynamicAlong(&[N]),
    );
    let projection = Projection::new(
        &[M, K, N],
        &[
            PhysicalAxisMap::affine(&[(M, 8), (K, 4)]).over(4),
            PhysicalAxisMap::of(N),
        ],
    );
    // Every coefficient divides by 4, so the division reduces away.
    assert_eq!(
        projection,
        Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::affine(&[(M, 2), (K, 1)]),
                PhysicalAxisMap::of(N)
            ],
        )
    );
    let _ = staged
        .arg(binding(&client, &[512, 64]), f32::elem_type_native())
        .gathered(projection)
        .build()
        .unwrap();
}

// ---- Launcher::vector_size -------------------------------------------------

#[test]
fn vector_size_picks_widest_qualifying_line() {
    let client = cubecl::test_device().client();
    // Everything divides: N's leaf edge is 32, both inner extents are 64.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    // Off real bindings, not hand-written dims: the strides are the allocator's, so a pitched
    // one that padded a row is what the gate sees.
    let rhs = Geometry::from(&binding(&client, &[16, 64]));
    let out = Geometry::from(&binding(&client, &[64, 64]));

    let v = launch.vector_size(N, &[(&rhs, &[K, N]), (&out, &[M, N])], size_of::<f32>());
    // The gate passed, so the pick is the hardware's widest line fitting the leaf edge (32).
    let expected = client
        .io_optimized_vector_sizes(size_of::<f32>())
        .filter(|&v| 32 % v == 0)
        .max()
        .unwrap_or(1);
    assert_eq!(v, expected);
    assert_eq!(32 % v, 0);
    assert_eq!(64 % v, 0);
}

#[test]
fn vector_size_falls_back_to_scalar() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let out = Geometry::from(&binding(&client, &[64, 64]));

    // An overhanging operand (k = 18 vs leaf 4) stays scalar: its masked accesses report
    // their length in lines and would wrongly clip.
    let overhang = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let rhs = Geometry::from(&binding(&client, &[18, 64]));
    assert_eq!(
        overhang.vector_size(N, &[(&rhs, &[K, N]), (&out, &[M, N])], size_of::<f32>()),
        1
    );

    // Col-major (innermost stride ≠ 1): lines wouldn't land on contiguous scalars.
    let col_major = Geometry::new(&[(16, 1), (64, 16)]);
    assert_eq!(
        launch.vector_size(
            N,
            &[(&col_major, &[K, N]), (&out, &[M, N])],
            size_of::<f32>()
        ),
        1
    );

    // An inner extent no width divides (63) blocks every line size.
    let odd = Geometry::from(&binding(&client, &[16, 63]));
    assert_eq!(
        launch.vector_size(N, &[(&odd, &[K, N])], size_of::<f32>()),
        1
    );
}

#[test]
fn arg_checked_and_vectorized_is_refused() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf, so the derived check is true: vectorizing must refuse.
    //
    // Two wide, not four: an axis overhanging a leaf of 4 has an extent 4 does not divide, so a
    // 4-wide line would be refused by `Geometry::serves_lines` first and this would stop probing
    // the bounds question. 18 is a whole number of 2-wide lines and still overhangs.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let refused = launch
        .arg(binding(&client, &[64, 18]), f32::elem_type_native())
        .axes(&[M, K])
        .vectorize(2)
        .build()
        .err();
    assert!(
        matches!(refused, Some(Refusal::UncheckableVectorEdge)),
        "{refused:?}"
    );
}

/// A contiguous dim the launch counts in blocks and a position inside one is in bounds as a single
/// axis would be: where only the rows overhang, it takes lines wider than one and no mask, and
/// the rows take the mask.
#[test]
fn arg_vectorized_along_a_blocked_dim_with_overhanging_rows_succeeds() {
    let client = cubecl::test_device().client();
    const KB: Axis = Axis(3);
    const KI: Axis = Axis(4);
    // M = 3 overhangs its leaf (8); K = 4 blocks of 16, whole.
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 3), (KB, 4), (KI, 16)]),
            cubecl_tile::Levels::leaf(&[(M, 8), (KB, 1), (KI, 16)])
                .walk_every(&[KB])
                .cubes(&[M])
                .build(),
        ),
        Form::Dynamic,
    );
    let blocked = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, 16), (KI, 1)]),
        ],
    );
    let arg = launch
        .arg(binding(&client, &[3, 64]), f32::elem_type_native())
        .gathered(blocked)
        .vectorize(4)
        .build()
        .unwrap();
    assert_eq!(
        arg.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );
}

#[test]
fn arg_vectorized_with_outer_axis_overhang_succeeds() {
    let client = cubecl::test_device().client();
    // M = 63 overhangs its leaf (8); N = 64 divides cleanly.
    let launch = {
        let (space, levels) = batched_space(1, 1, 63, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let arg = launch
        .arg(binding(&client, &[63, 64]), f32::elem_type_native())
        .axes(&[M, N])
        .vectorize(4)
        .build()
        .unwrap();
    assert!(arg.spec.is_checked());
    // M takes the mask, N carries the lines and needs none.
    assert_eq!(
        arg.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );
}

/// `checked(true)` states the mode, not the axis list: the per-axis derivation still keeps it off
/// the axes that cannot leave the buffer. Only `checked(false)` disarms outright.
#[test]
fn arg_explicit_check_still_narrows_to_the_unsettled_axes() {
    let client = cubecl::test_device().client();
    // k = 18 overhangs its leaf (4); M divides, so no override can put a mask on it.
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 18);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };

    let forced = launch
        .arg(binding(&client, &[64, 18]), f32::elem_type_native())
        .axes(&[M, K])
        .boundary(BoundaryPolicy::Every(Boundary::Zero))
        .build()
        .unwrap();
    assert_eq!(
        forced.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Zero)]
    );
}

/// The list is shaped over coordinate axes, and the setter is where a hand-built spec should learn
/// that: `Tile::of` would only catch it in-kernel, one comptime expansion away from the call.
#[test]
#[should_panic(expected = "2 coordinate axes")]
fn tile_spec_boundaries_must_match_the_coordinate_rank() {
    let _ = TileSpec::new(Projection::direct(&[M, N])).boundaries(&[None]);
}

#[test]
fn arg_gathered_clamp_vectorized_exemption() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(
            &client,
            Partitioning::new(space, levels),
            Form::DynamicAlong(&[N]),
        )
    };
    // 3 logical axes [M, K, N] mapped over 2 coordinate axes:
    // Spatial (M, K) on coordinate 0 (clamped), channel N on coordinate 1 (vectorized & in-bounds).
    let input = launch
        .arg(binding(&client, &[79, 64]), f32::elem_type_native())
        .gathered(window(1, 1, 0))
        .vectorize(4)
        .boundary(BoundaryPolicy::Every(Boundary::Clamp))
        .build()
        .unwrap();

    assert_eq!(
        input.spec.boundaries.as_slice(),
        &[Some(Boundary::Clamp), None]
    );
}

#[test]
#[should_panic(expected = "innermost dim")]
fn vector_size_axis_must_label_innermost() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(1, 1, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let lhs = binding(&client, &[64, 16]);
    // lhs's innermost dim is K, not N: asking for N-lines over it is a labeling bug.
    let _ = launch.vector_size(N, &[(&Geometry::from(&lhs), &[M, K])], size_of::<f32>());
}

#[test]
fn arg_more_batch_dims_than_axes_is_refused() {
    let client = cubecl::test_device().client();
    let launch = {
        let (space, levels) = batched_space(4, 3, 64, 64, 16);
        implied(&client, Partitioning::new(space, levels), Form::Dynamic)
    };
    let refused = launch
        .arg(binding(&client, &[4, 3, 64, 16]), f32::elem_type_native())
        .axes(&[M, K])
        .batches(&[B1])
        .build()
        .err();
    assert!(
        matches!(refused, Some(Refusal::UnlabelledBatchDims { .. })),
        "{refused:?}"
    );
}

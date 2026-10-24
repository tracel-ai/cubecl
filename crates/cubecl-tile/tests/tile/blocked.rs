//! An axis spelled as two: `x = xb · B + xi`, the axis partitioned into blocks.
//!
//! A *partition* of an axis is not a gather: the windows tile instead of overlapping, so a
//! logical position still determines a cell and every window is still a dense box. An operand
//! spanning `(KB, KI)` must therefore read exactly as one spanning `K` does, and cost the same.
//!
//! An axis exists when an operand *distinguishes* it: a quantized operand's scales vary over the
//! block index and not over the position inside the block, so those are two axes, which is why an
//! axis gets split at all.
//!
//! The first two tests split a *contracted* axis, without scales and then with them; the last two
//! split an axis the *output* spans, the shape a per-column-block scale needs and the one the
//! accumulator's edges had to be derived to allow.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;
use half::f16;

/// Which factor a test kernel writes its scales on; the engine has no such enum.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Scaled {
    Lhs,
    Rhs,
}

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contracted axis, unsplit: what the reference kernel walks.
const K: Axis = Axis(2);
/// Which block, and where inside it. Together they are `K`.
const KB: Axis = Axis(3);
const KI: Axis = Axis(4);
/// The same, for the output's columns: together they are `N`.
const NB: Axis = Axis(5);
const NI: Axis = Axis(6);

/// The register block every leaf here runs under.
const BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `C = A · B`, whichever axes the operands spell their contraction over.
#[cube(launch)]
fn matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c.at(&region).accumulating(BLOCK, Semiring::SUM_PROD);
        c_region.mma(&a.at(&region), &b.at(&region));
    }
}

/// `C = (A ⊗ S) · B`, the scales an operand like the other two.
#[cube(launch)]
fn scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c.at(&region).accumulating(BLOCK, Semiring::SUM_PROD);
        match comptime!(side) {
            Scaled::Lhs => c_region.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => c_region.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
}

/// The reference: one contracted axis, cut at the block. What the split has to reproduce.
#[test]
fn one_contracted_axis_is_the_reference() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (K, depth)]),
            Levels::leaf(&[(M, rows), (N, cols), (K, block)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth).map(|k| a[m * depth + k] * b[k * cols + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The same contraction with `K` spelled `(KB, KI)`, the walk stepping one block per region so the
/// leaf spans one. Same buffers, same numbers, and the direct nest either way.
#[test]
fn a_partitioned_axis_contracts_the_same() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth).map(|k| a[m * depth + k] * b[k * cols + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The split, with scales.** `S` carries `KI` and addresses nothing with it, so it cannot vary
/// inside a block. One scale per block stops being an arithmetic claim (`⌊k / 8⌋`) and becomes a
/// fact about which axes the operand spans, which is what already makes a broadcast a broadcast.
#[test]
fn scales_omit_the_axis_inside_the_block() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so the reference is exact.
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            // The block index alone. The position inside a block is not an axis of this operand,
            // which is the whole statement that one scale covers the block.
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

// A partition whose stride misses its block is refused where the projection and the nest first
// meet (`Projection::validate_composition`), unit-tested host-side in `physical::projection::base`
// rather than here: in the kernel it fires on a worker thread `#[should_panic]` never sees.

/// **An axis the output spans, split.** `N` is spelled `(NB, NI)` and the accumulator spans both,
/// so its column edge is two axes. Reading the trailing axis alone would take `NB` for the row
/// edge and contract against nothing; the edge instead spans every innermost axis not the lhs's.
#[test]
fn a_split_output_axis_contracts_the_same() {
    let (rows, blocks, inside, depth) = (4, 2, 4, 8);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth).map(|k| a[m * depth + k] * b[k * cols + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **A scale per column block, by omission.** The rhs's `N` is split, and the scales span `(NB,
/// NI)` addressing `NB` alone: one scale covers a whole block of columns because the operand has
/// no axis to vary over inside one, with nothing dividing anything.
///
/// This is the shape the rational spelling (`PhysicalAxisMap::of(N).over(bn)`) stands in for, and
/// the one that needed the accumulator's column edge to be a run: `NB` and `NI` are both the
/// rhs's, so both are columns.
#[test]
fn scales_omit_the_axis_inside_the_column_block() {
    let (rows, blocks, inside, depth) = (4, 2, 4, 8);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, so the reference is exact. One per column block.
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            // `K` carried and addressed by nothing, and the position inside a column block is
            // not an axis here at all: the scale varies over the block index alone.
            TileSpec::new(Projection::new(&[K, NB], &[PhysicalAxisMap::of(NB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Rhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[n / inside])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`a_split_output_axis_contracts_the_same`] with the rhs and the accumulator served as lines
/// exactly one column block wide, so the column group's two axes straddle the line boundary: `NI`
/// is one whole line and `NB` counts them.
#[cube(launch)]
fn wide_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c.at(&region).accumulating(BLOCK, Semiring::SUM_PROD);
        c_region.mma(&a.at(&region), &b.at(&region));
    }
}

#[test]
fn a_split_output_axis_serves_lines_one_block_wide() {
    let (rows, blocks, inside, depth) = (4, 2, 4, 8);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    wide_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            inside,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            inside,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth).map(|k| a[m * depth + k] * b[k * cols + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// `C = A · (B ⊗ S)` with the scales served as lines, as many a read as they are bound wide.
#[cube(launch)]
fn wide_scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c.at(&region).accumulating(BLOCK, Semiring::SUM_PROD);
        c_region.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region)));
    }
}

/// **The scales read as a line.** Their innermost axis is `NB`, the block index, an axis they
/// actually vary over, so a read of four serves four *different* scales covering four blocks of
/// columns. A value line's component is its ordinal along the shared edge, constant across the
/// block.
#[test]
fn scales_are_served_several_at_a_time() {
    let (rows, blocks, inside, depth, units) = (4, 4, 2, 8, 4);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Distinct per block, halves so the reference is exact.
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    wide_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            units,
            // The block index alone, and it is innermost: the width lands on the axis the scales
            // vary over, so one read serves `units` different scales.
            TileSpec::new(Projection::new(&[K, NB], &[PhysicalAxisMap::of(NB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[n / inside])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A split output axis against a *promoted* accumulator: the partials live in registers across the
/// whole walk, and the block they live in has to be sized by the column group rather than by its
/// innermost axis.
#[cube(launch)]
fn promoted_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    acc.drained_into(&c);
}

#[test]
fn a_promoted_accumulator_spans_a_split_output_axis() {
    let (rows, blocks, inside, depth) = (4, 4, 2, 8);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    promoted_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth).map(|k| a[m * depth + k] * b[k * cols + n]).sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`wide_scaled_matmul`] with the accumulator promoted to registers: the partials never touch
/// memory between `K` steps, which is what a decode step's one-row product keeps them in, and the scales covering
/// four column blocks still arrive in one read.
#[cube(launch)]
fn wide_scaled_promoted<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers { config: BLOCK }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level).unrolled() {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region)));
    }
    acc.drained_into(&c);
}

/// **The shape a decode step's one-row product runs.** Scales read as a line against a register accumulator.
#[test]
fn a_promoted_accumulator_takes_scales_by_the_line() {
    let (rows, blocks, inside, depth, units) = (4, 4, 2, 8, 4);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    wide_scaled_promoted::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            units,
            TileSpec::new(Projection::new(&[K, NB], &[PhysicalAxisMap::of(NB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[n / inside])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`wide_scaled_matmul`] with the scales at their own element type as well as their own width.
#[cube(launch)]
fn wide_typed_scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c.at(&region).accumulating(BLOCK, Semiring::SUM_PROD);
        c_region.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region)));
    }
}

/// **A scale is whatever its tensor holds, however many of them a read serves.** `f16` scales
/// against `f32` values, four to a line: the element type is the binding's and the width is the
/// binding's, and neither is the values'.
#[test]
fn scales_keep_their_own_element_when_served_as_lines() {
    let (rows, blocks, inside, depth, units) = (4, 4, 2, 8, 4);
    let cols = blocks * inside;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let scale_dtype = f16::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Halves, exact in f16 and in f32 alike.
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(scale_dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)]),
            Levels::leaf(&[(M, rows), (NB, blocks), (NI, inside), (K, depth)])
                .walk_every(&[M, NB, NI, K])
                .build(),
        ),
        Form::Static,
    );

    wide_typed_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[K, NB, NI],
                &[
                    PhysicalAxisMap::of(K),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            scale_dtype,
            units,
            TileSpec::new(Projection::new(&[K, NB], &[PhysicalAxisMap::of(NB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, NB, NI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(NB, inside), (NI, 1)]),
                ],
            )),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[n / inside])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

//! A kernel writes a [`TileTensor`] only through [`TileTensor::tile_mut`], which takes it by
//! `&mut`, so the launch's declaration of what it writes is its signature's and is true.
//!
//! A launch whose kernel panics as it expands runs nothing. The expansion runs on the device
//! thread, inside a task nobody waits on, so the only report of the panic is the claim the launch
//! leaves on the buffers it declares written, which fails their reads. With no compiled kernel to
//! say what it writes, the declaration is the signature's: a parameter taken by `&` is read-only.
//! A tile served by [`TileTensor::tile`] refuses a write as the kernel expands, so no kernel
//! writes an operand its launch leaves unclaimed, which a failed launch would hand back as the
//! result.

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_tile::*;

use crate::support::TileInput;

use super::uncut;

const ROW: Dimension = Dimension(0);
const EXTENT: usize = 4;
const REFUSAL: &str = "this tile kernel refuses to expand";

/// Panics as it expands when `refuse` says so, its output a written operand. Only the signature
/// matters: a kernel that never expands has no body to learn the rest from.
#[cube(launch)]
fn written_operand_refused(_out: &mut TileTensor<'_>, #[comptime] refuse: bool) {
    comptime!(assert!(!refuse, "{REFUSAL}"));
}

/// Zeroes `out` through the tile [`TileTensor::tile`] serves, which is read-only. `witness` is a
/// written operand, so the launch's decline fails its read.
#[cube(launch)]
fn zeroed_through_shared(out: &TileTensor<'_>, witness: &mut TileTensor<'_>, space: Partitioning) {
    let _witness = witness.tile_mut(&space);
    let mut out = out.tile(&space);
    out.zero();
}

/// Zeroes `out` through the tile [`TileTensor::tile_mut`] serves.
#[cube(launch)]
fn zeroed_through_mut(out: &mut TileTensor<'_>, space: Partitioning) {
    let mut out = out.tile_mut(&space);
    out.zero();
}

/// A row of [`EXTENT`] values counting up from zero, and a launch of one unit over its space.
fn row(client: &Client) -> (TileInput, Launcher) {
    let space = Space::new(&[(ROW, EXTENT)]);
    let launcher = uncut(client, &space, &space);
    (
        TileInput::builder(client, space).untiled().arange(),
        launcher,
    )
}

/// The report of the failed read of `handle`, which a launch refused before it ran left unwritten.
fn failed_read(client: &Client, handle: Handle) -> String {
    match client.read_one(handle) {
        Ok(bytes) => panic!(
            "a launch refused before it ran wrote nothing, yet the read returned {:?}",
            f32::from_bytes(&bytes)
        ),
        Err(error) => format!("{error:?}"),
    }
}

#[test]
fn a_written_operand_of_a_launch_refused_as_it_expands_fails_its_read() {
    let client = cubecl::test_device().client();
    let (out, launcher) = row(&client);

    written_operand_refused::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        out.arg(&launcher, 1),
        true,
    );

    let report = failed_read(&client, out.handle().handle);
    assert!(
        report.contains(REFUSAL),
        "the read must fail on the expansion's panic: {report}"
    );
}

#[test]
fn a_write_into_a_tile_served_through_a_shared_operand_is_refused_as_the_kernel_expands() {
    let client = cubecl::test_device().client();
    let (out, launcher) = row(&client);
    let (witness, _) = row(&client);

    zeroed_through_shared::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        out.arg(&launcher, 1),
        witness.arg(&launcher, 1),
        launcher.partitioning_arg(),
    );

    let report = failed_read(&client, witness.handle().handle);
    assert!(
        report.contains("is read-only") && report.contains("tile_mut"),
        "the launch must be refused for writing through `&`: {report}"
    );
}

#[test]
fn a_tile_served_through_a_written_operand_is_stored_into() {
    let client = cubecl::test_device().client();
    let (out, launcher) = row(&client);

    zeroed_through_mut::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        out.arg(&launcher, 1),
        launcher.partitioning_arg(),
    );

    let bytes = client
        .read_one(out.handle().handle)
        .expect("the launch wrote its output");
    assert_eq!(f32::from_bytes(&bytes), &[0.0; EXTENT]);
}

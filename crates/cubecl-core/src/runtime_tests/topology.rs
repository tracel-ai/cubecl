use crate as cubecl;
use alloc::vec::Vec;

use cubecl::prelude::*;

#[cube(launch, address_type = "dynamic")]
pub fn kernel_absolute_pos(output1: &mut [u32]) {
    if ABSOLUTE_POS >= output1.len() {
        terminate!();
    }

    output1[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
}

#[cube(launch, address_type = "dynamic")]
pub fn kernel_absolute_pos_cube(output1: &mut [u32]) {
    if ABSOLUTE_POS >= output1.len() {
        terminate!();
    }

    output1[ABSOLUTE_POS] = CUBE_POS as u32;
}

#[cube(launch)]
pub fn kernel_plane_pos(output: &mut [u32]) {
    let unit = UNIT_POS as usize;
    output[unit * 3] = PLANE_POS;
    output[unit * 3 + 1] = UNIT_POS_PLANE;
    output[unit * 3 + 2] = PLANE_DIM;
}

pub fn test_kernel_topology_absolute_pos(client: Client, addr_type: AddressType) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let cube_count = (3, 5, 7);
    let cube_dim = (2, 2, 1);

    let length = cube_count.0 * cube_count.1 * cube_count.2 * cube_dim.0 * cube_dim.1 * cube_dim.2;
    let handle1 = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            cube_dim.into(),
            addr_type,
            BufferArg::from_raw_parts(handle1.clone(), length as usize),
        )
    };

    let actual = client.read_one_unchecked(handle1);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..length).collect();

    assert_eq!(actual, &expect);
}

/// `ABSOLUTE_POS` is cube major: one cube's units occupy one contiguous run.
///
/// The bijectivity the test above checks holds under any ordering, so it cannot
/// see a cube whose units are scattered across the grid.
pub fn test_kernel_topology_absolute_pos_is_cube_major(client: Client, addr_type: AddressType) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let cube_count = (3, 5, 7);
    let cube_dim = (2, 2, 1);

    let units_per_cube = cube_dim.0 * cube_dim.1 * cube_dim.2;
    let cubes = cube_count.0 * cube_count.1 * cube_count.2;
    let length = cubes * units_per_cube;
    let handle = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos_cube::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, cube_count.2),
            cube_dim.into(),
            addr_type,
            BufferArg::from_raw_parts(handle.clone(), length as usize),
        )
    };

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..cubes)
        .flat_map(|cube| core::iter::repeat_n(cube, units_per_cube as usize))
        .collect();

    assert_eq!(actual, &expect);
}

/// `PLANE_POS` is the plane a unit belongs to, planes being consecutive runs of `PLANE_DIM`
/// units in `UNIT_POS` order: each unit is lane `UNIT_POS_PLANE` of plane `PLANE_POS`. Read on a
/// two-dimensional cube of at least four planes of whatever width the kernel runs at, so the unit
/// order the planes follow is the linearized one, and the lane the hardware reports checks the
/// plane a target derives.
pub fn test_kernel_topology_plane_pos(client: Client) {
    let hardware = &client.properties().hardware;
    let (min_dim, max_dim) = (hardware.plane_size_min, hardware.plane_size_max);
    let cube_dim = CubeDim::new_2d(max_dim / 2, 8);
    let units = cube_dim.num_elems() as usize;
    let handle = client.empty(units * 3 * core::mem::size_of::<u32>());

    unsafe {
        kernel_plane_pos::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            cube_dim,
            BufferArg::from_raw_parts(handle.clone(), units * 3),
        )
    };

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    for unit in 0..units {
        let (plane, lane, dim) = (actual[unit * 3], actual[unit * 3 + 1], actual[unit * 3 + 2]);
        assert!(
            (min_dim..=max_dim).contains(&dim),
            "unit {unit}: PLANE_DIM {dim} outside the device's {min_dim}..={max_dim}"
        );
        assert_eq!(
            (plane, lane),
            (unit as u32 / dim, unit as u32 % dim),
            "unit {unit}: (PLANE_POS, UNIT_POS_PLANE)"
        );
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_topology {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_topology_scalar() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos(
                client.clone(),
                AddressType::U32,
            );
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos(
                client,
                AddressType::U64,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_topology_absolute_pos_is_cube_major() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos_is_cube_major(
                client.clone(),
                AddressType::U32,
            );
            cubecl_core::runtime_tests::topology::test_kernel_topology_absolute_pos_is_cube_major(
                client,
                AddressType::U64,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_topology_plane_pos() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_plane_pos(client);
        }
    };
}

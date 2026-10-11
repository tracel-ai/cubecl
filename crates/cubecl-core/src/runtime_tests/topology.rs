use crate as cubecl;
use alloc::vec::Vec;

use cubecl::prelude::*;
use cubecl_ir::features::Plane;

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

/// Each unit's axes, its plane, lane and plane width, and its absolute position along x and y,
/// at its cube's run of the output.
#[cube(launch)]
pub fn kernel_plane_pos_2d(output: &mut [u32]) {
    let at = (CUBE_POS * CUBE_DIM as usize + UNIT_POS as usize) * 7;
    output[at] = UNIT_POS_X;
    output[at + 1] = UNIT_POS_Y;
    output[at + 2] = PLANE_POS;
    output[at + 3] = UNIT_POS_PLANE;
    output[at + 4] = PLANE_DIM;
    output[at + 5] = ABSOLUTE_POS_X;
    output[at + 6] = ABSOLUTE_POS_Y;
}

/// Each unit's axes and linear position, and its absolute position along each axis and in all,
/// at its cube's run of the output.
#[cube(launch)]
pub fn kernel_unit_pos_3d(output: &mut [u32]) {
    let at = (CUBE_POS * CUBE_DIM as usize + UNIT_POS as usize) * 8;
    output[at] = UNIT_POS_X;
    output[at + 1] = UNIT_POS_Y;
    output[at + 2] = UNIT_POS_Z;
    output[at + 3] = UNIT_POS;
    output[at + 4] = ABSOLUTE_POS_X;
    output[at + 5] = ABSOLUTE_POS_Y;
    output[at + 6] = ABSOLUTE_POS_Z;
    output[at + 7] = CUBE_POS as u32;
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
/// cube of at least four planes of whatever width the kernel runs at, so the lane the hardware
/// reports checks the plane a target derives. One-dimensional, as WGSL takes its subgroup
/// builtins only in a one-dimensional workgroup; a device without plane operations has no planes
/// to place a unit in, and is skipped.
pub fn test_kernel_topology_plane_pos(client: Client) {
    if !client.features().plane.contains(Plane::Ops) {
        return;
    }
    let hardware = &client.properties().hardware;
    let (min_dim, max_dim) = (hardware.plane_size_min, hardware.plane_size_max);
    let cube_dim = CubeDim::new_1d(max_dim * 4);
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

/// [`test_kernel_topology_plane_pos`] on a cube spanning `y` as well, a plane's width of units
/// along `x` and four along `y`: a unit's axes are its linear position's digits, its plane and
/// lane are that position's, and its absolute position along an axis is its cube's times the
/// cube's extent there, plus its own. WGSL takes its subgroup builtins only in a one-dimensional
/// workgroup, so it launches such a cube flat and is held to those planes. Elsewhere the cube
/// keeps its shape and the API leaves which units share a plane to the driver, so there its
/// planes need only split the cube into whole planes of distinct lanes.
pub fn test_kernel_topology_plane_pos_in_a_cube_spanning_y(client: Client) {
    if !client.features().plane.contains(Plane::Ops) {
        return;
    }
    let width = client.properties().hardware.plane_size_max;
    let (cube_dim, cube_count) = (CubeDim::new_2d(width, 4), (2u32, 3u32));
    if cube_dim.num_elems() > client.properties().hardware.max_units_per_cube {
        return;
    }
    let launches_flat = client.name() == "wgpu<wgsl>";
    let units = cube_dim.num_elems() as usize;
    let cubes = (cube_count.0 * cube_count.1) as usize;
    let handle = client.empty(cubes * units * 7 * core::mem::size_of::<u32>());

    unsafe {
        kernel_plane_pos_2d::launch(
            &client,
            CubeCount::Static(cube_count.0, cube_count.1, 1),
            cube_dim,
            BufferArg::from_raw_parts(handle.clone(), cubes * units * 7),
        )
    };

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual.len(), cubes * units * 7, "the launch ran");
    for cube in 0..cubes as u32 {
        let (cube_x, cube_y) = (cube % cube_count.0, cube / cube_count.0);
        let mut lanes = Vec::new();
        for unit in 0..units as u32 {
            let at = |field: usize| actual[(cube as usize * units + unit as usize) * 7 + field];
            let (x, y) = (unit % width, unit / width);
            assert_eq!(
                (at(0), at(1)),
                (x, y),
                "unit {unit}: (UNIT_POS_X, UNIT_POS_Y)"
            );
            let dim = at(4);
            if launches_flat {
                assert_eq!(
                    (at(2), at(3)),
                    (unit / dim, unit % dim),
                    "unit {unit}: (PLANE_POS, UNIT_POS_PLANE) at PLANE_DIM {dim}"
                );
            }
            lanes.push(at(2) * dim + at(3));
            assert_eq!(
                (at(5), at(6)),
                (cube_x * width + x, cube_y * 4 + y),
                "cube {cube} unit {unit}: (ABSOLUTE_POS_X, ABSOLUTE_POS_Y)"
            );
        }
        lanes.sort_unstable();
        assert_eq!(
            lanes,
            (0..units as u32).collect::<Vec<_>>(),
            "cube {cube}: every (PLANE_POS, UNIT_POS_PLANE) a distinct lane of a whole plane"
        );
    }
}

/// A cube spanning all three axes, three units wide so an axis is a real division of the linear
/// position: a unit's axes are that position's digits, `x` fastest, and its absolute position
/// along an axis is its cube's times the cube's extent there, plus its own — however a backend's
/// compiler lays the workgroup out.
pub fn test_kernel_topology_unit_pos_in_a_cube_spanning_z(client: Client) {
    let (dim, count) = ((3u32, 2u32, 2u32), (2u32, 1u32, 2u32));
    let cube_dim = CubeDim::new_3d(dim.0, dim.1, dim.2);
    let units = cube_dim.num_elems() as usize;
    let cubes = (count.0 * count.1 * count.2) as usize;
    let handle = client.empty(cubes * units * 8 * core::mem::size_of::<u32>());

    unsafe {
        kernel_unit_pos_3d::launch(
            &client,
            CubeCount::Static(count.0, count.1, count.2),
            cube_dim,
            BufferArg::from_raw_parts(handle.clone(), cubes * units * 8),
        )
    };

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual.len(), cubes * units * 8, "the launch ran");
    for cube in 0..cubes as u32 {
        let (cube_x, cube_y, cube_z) = (
            cube % count.0,
            (cube / count.0) % count.1,
            cube / (count.0 * count.1),
        );
        for unit in 0..units as u32 {
            let at = |field: usize| actual[(cube as usize * units + unit as usize) * 8 + field];
            let (x, y, z) = (unit % dim.0, (unit / dim.0) % dim.1, unit / (dim.0 * dim.1));
            assert_eq!(
                (at(0), at(1), at(2), at(3)),
                (x, y, z, unit),
                "unit {unit}: (UNIT_POS_X, UNIT_POS_Y, UNIT_POS_Z, UNIT_POS)"
            );
            assert_eq!(
                (at(4), at(5), at(6), at(7)),
                (
                    cube_x * dim.0 + x,
                    cube_y * dim.1 + y,
                    cube_z * dim.2 + z,
                    cube
                ),
                "cube {cube} unit {unit}: (ABSOLUTE_POS_X/Y/Z, CUBE_POS)"
            );
        }
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
        #[$crate::runtime_tests::test_log::test]
        fn test_topology_plane_pos_in_a_cube_spanning_y() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_plane_pos_in_a_cube_spanning_y(
                client,
            );
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_topology_unit_pos_in_a_cube_spanning_z() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::topology::test_kernel_topology_unit_pos_in_a_cube_spanning_z(
                client,
            );
        }
    };
}

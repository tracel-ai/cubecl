use alloc::{vec, vec::Vec};
use cubecl_runtime::runtime::Runtime;

use crate::prelude::*;
use crate::{self as cubecl};
use cubecl_ir::features::{AtomicUsage, Plane};

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_sync_cube(buffer: &mut [u32], out: &mut [u32]) {
    let unit_pos = UNIT_POS as usize;
    buffer[unit_pos] = UNIT_POS;
    sync_cube();
    if unit_pos != 0 {
        out[unit_pos] = buffer[unit_pos - 1] + buffer[unit_pos];
    }
}

pub fn test_sync_cube<R: Runtime>(client: Client) {
    // Clamp the cube dim to the device's limit (e.g. the core count on CPU).
    let max_units = client.properties().hardware.max_units_per_cube;
    let dim_x = core::cmp::min(8, (max_units / 2).max(1));
    let units = (dim_x * 2) as usize;

    let handle = client.empty(32 * core::mem::size_of::<u32>());
    let test = client.empty(32 * core::mem::size_of::<u32>());

    kernel_test_sync_cube::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(dim_x, 2),
        unsafe { BufferArg::from_raw_parts(test, 32) },
        unsafe { BufferArg::from_raw_parts(handle.clone(), 32) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    let expected: Vec<u32> = (0..units as i32)
        .map(|i| core::cmp::max(2 * i - 1, 0) as u32)
        .collect();

    assert_eq!(&actual[1..units], &expected[1..units]);
}

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_finished_sync_cube(buffer: &mut [u32], out: &mut [u32]) {
    let unit_pos = UNIT_POS as usize;
    buffer[unit_pos] = UNIT_POS;
    if UNIT_POS > 16 {
        terminate!();
    }
    sync_cube();
    sync_cube();
    if UNIT_POS != 0 {
        out[unit_pos] = buffer[unit_pos - 1] + buffer[unit_pos];
    }
    sync_cube();
}

pub fn test_finished_sync_cube<R: Runtime>(client: Client) {
    // Clamp the cube dim to the device's limit (e.g. the core count on CPU).
    let max_units = client.properties().hardware.max_units_per_cube;
    let dim_x = core::cmp::min(8, (max_units / 2).max(1));
    let checked = core::cmp::min(8, (dim_x * 2) as usize);

    let handle = client.empty(32 * core::mem::size_of::<u32>());
    let test = client.empty(32 * core::mem::size_of::<u32>());

    kernel_test_finished_sync_cube::launch(
        &client,
        CubeCount::Static(2, 1, 1),
        CubeDim::new_2d(dim_x, 2),
        unsafe { BufferArg::from_raw_parts(test, 32) },
        unsafe { BufferArg::from_raw_parts(handle.clone(), 32) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    let expected: Vec<u32> = (0..checked as i32)
        .map(|i| core::cmp::max(2 * i - 1, 0) as u32)
        .collect();

    assert_eq!(&actual[1..checked], &expected[1..checked]);
}

#[cube(launch)]
/// First 32 elements should be 1, while last 32 elements may or may not be 1
fn kernel_test_sync_plane<F: Float>(out: &mut [F]) {
    let mut shared_memory = Shared::<F>::new();

    if UNIT_POS == 0 {
        *shared_memory = F::from_int(1);
    }

    sync_plane();

    out[UNIT_POS as usize] = *shared_memory;
}

pub fn test_sync_plane<R: Runtime>(client: Client) {
    if !client.features().plane.contains(Plane::Sync) {
        // We can't execute the test, skip.
        return;
    }

    let handle = client.empty(64 * core::mem::size_of::<f32>());

    kernel_test_sync_plane::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(32, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 64) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);
    let expected = &[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    assert_eq!(&actual[0..32], expected);
}

#[cube(launch)]
/// All 64 elements should be 1
fn kernel_test_sync_cube_shared<F: Float>(out: &mut [F]) {
    let mut shared_memory = Shared::<F>::new();

    if UNIT_POS == 0 {
        *shared_memory = F::from_int(1);
    }

    sync_cube();

    out[UNIT_POS as usize] = *shared_memory;
}

pub fn test_sync_cube_shared<R: Runtime>(client: Client) {
    let max_cube_count = std::cmp::min(64, client.properties().hardware.max_units_per_cube);
    let handle = client.empty(max_cube_count as usize * core::mem::size_of::<f32>());

    kernel_test_sync_cube_shared::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(max_cube_count / 2, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), max_cube_count as usize) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);
    let expected = vec![1.0; max_cube_count as usize];

    assert_eq!(&actual[0..max_cube_count as usize], expected);
}

#[cube(launch)]
fn kernel_test_workgroup_uniform_load(out: &mut [u32]) {
    let mut count = Shared::new_slice(1usize);
    if UNIT_POS == 0 {
        count[0] = 3u32;
    }
    sync_cube();

    let n = workgroup_uniform_load(&count[0]);
    if n > 0 {
        sync_cube();
    }
    out[UNIT_POS as usize] = n;
}

pub fn test_workgroup_uniform_load<R: Runtime>(client: Client) {
    let max_cube_count = std::cmp::min(64, client.properties().hardware.max_units_per_cube);
    let handle = client.empty(max_cube_count as usize * core::mem::size_of::<u32>());

    kernel_test_workgroup_uniform_load::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(max_cube_count / 2, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), max_cube_count as usize) },
    );

    let actual = client.read_one_unchecked(handle);
    let expected = vec![3u32; max_cube_count as usize];
    assert_eq!(u32::from_bytes(&actual), &expected);
}

#[cube(launch)]
fn kernel_test_workgroup_uniform_load_atomic(out: &mut [u32]) {
    let count = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if UNIT_POS == 0 {
        count[0].store(3u32);
    }
    sync_cube();

    let n = workgroup_uniform_load_atomic(&count[0]);
    if n > 0 {
        sync_cube();
    }
    out[UNIT_POS as usize] = n;
}

pub fn test_workgroup_uniform_load_atomic<R: Runtime>(client: Client) {
    let ty = Type::atomic(u32::elem_type_native());
    if !client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::LoadStore)
    {
        return;
    }

    let max_cube_count = std::cmp::min(64, client.properties().hardware.max_units_per_cube);
    let handle = client.empty(max_cube_count as usize * core::mem::size_of::<u32>());

    kernel_test_workgroup_uniform_load_atomic::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(max_cube_count / 2, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), max_cube_count as usize) },
    );

    let actual = client.read_one_unchecked(handle);
    let expected = vec![3u32; max_cube_count as usize];
    assert_eq!(u32::from_bytes(&actual), &expected);
}

#[cube(launch)]
fn kernel_test_workgroup_uniform_load_vec<N: Size>(out: &mut [Vector<f32, N>]) {
    let mut smem = Shared::new_slice(1usize);
    if UNIT_POS == 0 {
        smem[0] = Vector::new(7.0f32);
    }
    sync_cube();

    out[UNIT_POS as usize] = workgroup_uniform_load(&smem[0]);
}

pub fn test_workgroup_uniform_load_vec<R: Runtime>(client: Client) {
    let lanes = 4usize;
    let max_cube_count = std::cmp::min(64, client.properties().hardware.max_units_per_cube);
    let output = client.create_from_slice(f32::as_bytes(&vec![
        0.0f32;
        max_cube_count as usize * lanes
    ]));

    kernel_test_workgroup_uniform_load_vec::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(max_cube_count / 2, 2),
        lanes,
        unsafe { BufferArg::from_raw_parts(output.clone(), 64) },
    );

    let actual = client.read_one_unchecked(output);
    assert!(f32::from_bytes(&actual).iter().all(|&x| x == 7.0f32));
}

/// `workgroup_uniform_load` has to synchronise on its own: one unit publishes a
/// value and the rest read it back through the uniform load, with no explicit
/// `sync_cube` in between. The value is used as a loop bound, so a stale read
/// changes how much work the reader does rather than just its output.
#[cube(launch)]
fn kernel_test_workgroup_uniform_load_synchronizes(out: &mut [u32]) {
    let mut bound = Shared::new_slice(1usize);
    if UNIT_POS == 0 {
        // Derive the bound from real work so it cannot be constant-folded and
        // the store cannot be hoisted above the readers.
        let mut acc = 0u32;
        let mut i = 0u32;
        while i < 1024u32 {
            acc += i % 3u32;
            i += 1u32;
        }
        bound[0] = acc;
    }
    let n = workgroup_uniform_load(&bound[0]);
    let capped = min(n, 4096u32);
    let mut sum = 0u32;
    let mut k = 0u32;
    while k < capped {
        sum += k;
        k += 1u32;
    }
    out[UNIT_POS as usize] = sum;
}

fn expected_uniform_load_sum() -> u32 {
    let acc: u32 = (0..1024u32).map(|i| i % 3).sum();
    let capped = acc.min(4096);
    (0..capped).sum()
}

pub fn test_workgroup_uniform_load_synchronizes<R: Runtime>(client: Client) {
    let units = std::cmp::min(256, client.properties().hardware.max_units_per_cube);
    let handle = client.empty(units as usize * core::mem::size_of::<u32>());

    kernel_test_workgroup_uniform_load_synchronizes::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(units / 2, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), units as usize) },
    );

    let actual = client.read_one_unchecked(handle);
    let expected = vec![expected_uniform_load_sum(); units as usize];
    assert_eq!(u32::from_bytes(&actual), &expected);
}

/// Atomic counterpart of [`test_workgroup_uniform_load_synchronizes`].
#[cube(launch)]
fn kernel_test_workgroup_uniform_load_atomic_synchronizes(out: &mut [u32]) {
    let bound = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if UNIT_POS == 0 {
        let mut acc = 0u32;
        let mut i = 0u32;
        while i < 1024u32 {
            acc += i % 3u32;
            i += 1u32;
        }
        Atomic::store(&bound[0], acc);
    }
    let n = workgroup_uniform_load_atomic(&bound[0]);
    let capped = min(n, 4096u32);
    let mut sum = 0u32;
    let mut k = 0u32;
    while k < capped {
        sum += k;
        k += 1u32;
    }
    out[UNIT_POS as usize] = sum;
}

pub fn test_workgroup_uniform_load_atomic_synchronizes<R: Runtime>(client: Client) {
    let units = std::cmp::min(256, client.properties().hardware.max_units_per_cube);
    let handle = client.empty(units as usize * core::mem::size_of::<u32>());

    kernel_test_workgroup_uniform_load_atomic_synchronizes::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(units / 2, 2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), units as usize) },
    );

    let actual = client.read_one_unchecked(handle);
    let expected = vec![expected_uniform_load_sum(); units as usize];
    assert_eq!(u32::from_bytes(&actual), &expected);
}

/// One cube reads what the others published, with no second dispatch: every cube reduces its
/// units to one partial, releases it with [`sync_storage`], and announces itself on a counter.
/// The cube whose arrival is the last acquires the rest and sums them.
///
/// Both halves of the scope are under test, and the shape is chosen so that each of them has to
/// work. The partial is written by *one* unit and is a reduction over all of them, so the
/// release has to cover a store the cube made together rather than one store per unit. The count
/// is taken by one unit and read by every one of them, which is the cube barrier. Nothing spins
/// — the cubes that are not last simply end — so this cannot hang on a device that does not run
/// them all at once.
///
/// It catches a lowering with no device release at all, which is what CUDA and Metal both had.
/// It does *not* catch one whose release sits on the wrong side of the barrier: on an M2 this
/// kernel answers correctly under that too, and the case that does not is a longer one, in
/// cubek's `the_last_cube_in_merges_the_others`.
#[cube(launch)]
fn kernel_test_sync_storage_across_cubes(
    partials: &mut [u32],
    counter: &mut [Atomic<u32>],
    out: &mut [u32],
    #[comptime] cubes: u32,
    #[comptime] units: u32,
) {
    let mut mine = Shared::<[u32]>::new_slice(units as usize);
    mine[UNIT_POS as usize] = CUBE_POS as u32 * units + UNIT_POS + 1;
    sync_cube();
    if UNIT_POS == 0 {
        let mut total = 0u32;
        let mut i = 0u32;
        while i < units {
            total += mine[i as usize];
            i += 1u32;
        }
        partials[CUBE_POS] = total;
    }

    let mut arrived = Shared::<u32>::new();
    // Release: the partial this cube just published is visible to whichever cube is last.
    sync_storage();
    if UNIT_POS == 0 {
        *arrived = counter[0].fetch_add(1u32);
    }
    // Acquire, and the cube half of the same scope: the count reaches every unit, and what the
    // cubes that arrived before published is visible to this one.
    sync_storage();

    if *arrived == cubes - 1 {
        // A stripe per unit, so the count has to have reached all of them.
        let mut sum = 0u32;
        let mut cube = UNIT_POS;
        while cube < cubes {
            sum += partials[cube as usize];
            cube += units;
        }
        out[UNIT_POS as usize] = sum;
    }
}

pub fn test_sync_storage_across_cubes<R: Runtime>(client: Client) {
    if !client.properties().features.device_memory_scope {
        // The runtime does not promise that one cube's writes reach another, so say so rather
        // than pass silently.
        std::println!("device memory scope not supported - skipped");
        return;
    }
    let ty = Type::atomic(u32::elem_type_native());
    if !client
        .properties()
        .atomic_type_usage(ty)
        .contains(AtomicUsage::Add)
    {
        std::println!("u32 atomic add not supported - skipped");
        return;
    }

    let cubes = 32u32;
    let units = core::cmp::min(32, client.properties().hardware.max_units_per_cube);

    let partials = client.empty(cubes as usize * core::mem::size_of::<u32>());
    let counter = client.create_from_slice(u32::as_bytes(&[0u32]));
    let out = client.create_from_slice(u32::as_bytes(&vec![0u32; units as usize]));

    kernel_test_sync_storage_across_cubes::launch(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(partials, cubes as usize) },
        unsafe { BufferArg::from_raw_parts(counter, 1) },
        unsafe { BufferArg::from_raw_parts(out.clone(), units as usize) },
        cubes,
        units,
    );

    let partial = |cube: u32| (0..units).map(|unit| cube * units + unit + 1).sum::<u32>();
    let expected: Vec<u32> = (0..units)
        .map(|unit| (unit..cubes).step_by(units as usize).map(partial).sum())
        .collect();

    let actual = client.read_one_unchecked(out);
    assert_eq!(u32::from_bytes(&actual), &expected);
}

#[macro_export]
macro_rules! testgen_sync_plane {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_plane() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_plane::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_cube() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_cube::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_finished_sync_cube() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_finished_sync_cube::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_storage_across_cubes() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_storage_across_cubes::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_sync_cube_shared() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_sync_cube_shared::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_workgroup_uniform_load_synchronizes() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_workgroup_uniform_load_synchronizes::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_workgroup_uniform_load_atomic_synchronizes() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_workgroup_uniform_load_atomic_synchronizes::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_workgroup_uniform_load() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_workgroup_uniform_load::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_workgroup_uniform_load_atomic() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_workgroup_uniform_load_atomic::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_workgroup_uniform_load_vec() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::synchronization::test_workgroup_uniform_load_vec::<
                TestRuntime,
            >(client);
        }
    };
}

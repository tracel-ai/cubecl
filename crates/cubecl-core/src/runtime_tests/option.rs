use crate as cubecl;
use cubecl::prelude::*;

// SCALAR

#[cube(launch)]
pub fn kernel_option_scalar(array: &mut Array<i32>, value: Option<i32>) {
    match comptime!(value) {
        Some(value) => array[0] = value,
        None => {}
    }
}

pub fn test_option_scalar_none<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5]));

    unsafe {
        kernel_option_scalar::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 1, 1),
            None,
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 5);
}

pub fn test_option_scalar_some<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5]));

    unsafe {
        kernel_option_scalar::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 1, 1),
            Some(ScalarArg::new(10)),
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual[0], 10);
}

// ARRAY

#[cube(launch)]
pub fn kernel_option_array(array: &mut Array<i32>, data: &Option<Array<i32>>) {
    match comptime!(data) {
        Some(data) => array[UNIT_POS] = data[UNIT_POS],
        None => {}
    }
}

//
pub fn test_option_array_none<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5, 5]));

    unsafe {
        kernel_option_array::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 2, 1),
            None,
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [5, 5]);
}

pub fn test_option_array_some<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5, 5]));
    let data = client.create(i32::as_bytes(&[10, 20]));

    unsafe {
        kernel_option_array::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 2, 1),
            Some(ArrayArg::from_raw_parts::<i32>(&data, 2, 1)),
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [10, 20]);
}

// MAP

#[cube(launch)]
pub fn kernel_option_map(array: &mut Array<i32>, value: Option<i32>) {
    let memory = match comptime!(value) {
        Some(value) => {
            let mut m = SharedMemory::new(2);
            m[UNIT_POS] = value;
            some::<SharedMemory<i32>>(m)
        }
        None => None,
    };
    let slice = match comptime!(memory) {
        Some(memory) => some::<Slice<i32>>(memory.to_slice()),
        None => None,
    };
    option_map(array.to_slice_mut(), slice);
}

#[cube]
fn option_map(mut output: SliceMut<i32>, data: Option<Slice<i32>>) {
    match comptime!(data) {
        Some(data) => output[UNIT_POS] = data[UNIT_POS],
        None => {}
    }
}

pub fn test_option_map_none<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5, 5]));

    unsafe {
        kernel_option_map::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 2, 1),
            None,
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [5, 5]);
}

pub fn test_option_map_some<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let array = client.create(i32::as_bytes(&[5, 5]));

    unsafe {
        kernel_option_map::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(2, 1, 1),
            ArrayArg::from_raw_parts::<i32>(&array, 2, 1),
            Some(ScalarArg::new(10)),
        )
    };

    let actual = client.read_one(array.binding());
    let actual = i32::from_bytes(&actual);

    assert_eq!(actual, [10, 10]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_option {
    () => {
        use super::*;

        #[test]
        fn test_option_scalar_none() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_scalar_none::<TestRuntime>(client);
        }

        #[test]
        fn test_option_scalar_some() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_scalar_some::<TestRuntime>(client);
        }

        #[test]
        fn test_option_array_none() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_array_none::<TestRuntime>(client);
        }

        #[test]
        fn test_option_array_some() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_array_some::<TestRuntime>(client);
        }

        #[test]
        fn test_option_map_none() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_map_none::<TestRuntime>(client);
        }

        #[test]
        fn test_option_map_some() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::option::test_option_map_some::<TestRuntime>(client);
        }
    };
}

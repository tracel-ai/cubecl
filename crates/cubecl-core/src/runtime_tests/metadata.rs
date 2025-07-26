use crate as cubecl;

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn kernel_shape_dim_4(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[0] = lhs.shape(0);
    out[1] = lhs.shape(1);
    out[2] = lhs.shape(2);
    out[3] = lhs.shape(3);
    out[4] = rhs.shape(0);
    out[5] = rhs.shape(1);
    out[6] = rhs.shape(2);
    out[7] = rhs.shape(3);
    out[8] = out.shape(0);
    out[9] = out.shape(1);
    out[10] = out.shape(2);
    out[11] = out.shape(3);
}

#[cube(launch_unchecked)]
pub fn kernel_shape_different_ranks(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[0] = lhs.shape(0);
    out[1] = lhs.shape(1);
    out[2] = lhs.shape(2);
    out[3] = lhs.shape(3);
    out[4] = rhs.shape(0);
    out[5] = rhs.shape(1);
    out[6] = rhs.shape(2);
    out[7] = out.shape(0);
    out[8] = out.shape(1);
    out[9] = lhs.rank();
    out[10] = rhs.rank();
    out[11] = out.rank();
}

#[cube(launch_unchecked)]
pub fn kernel_stride_different_ranks(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[0] = lhs.stride(0);
    out[1] = lhs.stride(1);
    out[2] = lhs.stride(2);
    out[3] = lhs.stride(3);
    out[4] = rhs.stride(0);
    out[5] = rhs.stride(1);
    out[6] = rhs.stride(2);
    out[7] = out.stride(0);
    out[8] = out.stride(1);
}

#[cube(launch_unchecked)]
pub fn kernel_len_different_ranks(lhs: &Tensor<f32>, rhs: &Tensor<f32>, out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[0] = lhs.len();
    out[1] = rhs.len();
    out[2] = out.len();
}

#[cube(launch_unchecked)]
pub fn kernel_buffer_len(out: &mut Tensor<u32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }

    out[0] = out.buffer_len();
}

pub fn test_shape_dim_4<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle2 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle3 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_shape_dim_4::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[1, 1, 1, 1], &[2, 3, 4, 5], 1),
            TensorArg::from_raw_parts::<u32>(&handle2, &[1, 1, 1, 1], &[9, 8, 7, 6], 1),
            TensorArg::from_raw_parts::<u32>(&handle3, &[1, 1, 1, 1], &[10, 11, 12, 13], 1),
        )
    };

    let actual = client.read_one(handle3.binding());
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = vec![2, 3, 4, 5, 9, 8, 7, 6, 10, 11, 12, 13];

    assert_eq!(actual, &expect);
}

pub fn test_shape_different_ranks<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle2 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle3 = client
        .empty(12 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_shape_different_ranks::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[1, 1, 1, 1], &[2, 3, 4, 5], 1),
            TensorArg::from_raw_parts::<u32>(&handle2, &[1, 1, 1], &[9, 8, 7], 1),
            TensorArg::from_raw_parts::<u32>(&handle3, &[1, 1], &[10, 11], 1),
        )
    };

    let actual = client.read_one(handle3.binding());
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = vec![2, 3, 4, 5, 9, 8, 7, 10, 11, 4, 3, 2];

    assert_eq!(actual, &expect);
}

pub fn test_stride_different_ranks<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(9 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle2 = client
        .empty(9 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle3 = client
        .empty(9 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_stride_different_ranks::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[1, 2, 3, 4], &[1, 1, 1, 1], 1),
            TensorArg::from_raw_parts::<u32>(&handle2, &[4, 5, 6], &[1, 1, 1], 1),
            TensorArg::from_raw_parts::<u32>(&handle3, &[3, 2], &[1, 1], 1),
        )
    };

    let actual = client.read_one(handle3.binding());
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = vec![1, 2, 3, 4, 4, 5, 6, 3, 2];

    assert_eq!(actual, &expect);
}

pub fn test_len_different_ranks<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(3 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle2 = client
        .empty(3 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    let handle3 = client
        .empty(3 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_len_different_ranks::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[1, 1, 1, 1], &[2, 3, 4, 5], 1),
            TensorArg::from_raw_parts::<u32>(&handle2, &[1, 1, 1], &[9, 8, 7], 1),
            TensorArg::from_raw_parts::<u32>(&handle3, &[1, 1], &[10, 11], 1),
        )
    };

    let actual = client.read_one(handle3.binding());
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = vec![2 * 3 * 4 * 5, 9 * 8 * 7, 10 * 11];

    assert_eq!(actual, &expect);
}

pub fn test_buffer_len_discontiguous<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(64 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_buffer_len::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[32, 16, 4, 1], &[2, 2, 2, 2], 1),
        )
    };

    let actual = client.read_one(handle1.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual[0], 64);
}

pub fn test_buffer_len_vectorized<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(32 * core::mem::size_of::<u32>())
        .expect("Alloc failed");

    unsafe {
        kernel_buffer_len::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[16, 8, 4, 1], &[2, 2, 2, 4], 4),
        )
    };

    let actual = client.read_one(handle1.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual[0], 8);
}

pub fn test_buffer_len_offset<R: Runtime>(client: ComputeClient<R::Server, R::Channel>) {
    let handle1 = client
        .empty(256 * core::mem::size_of::<u32>())
        .expect("Alloc failed");
    // We use an offset of 256 bytes here because this is the default in WebGPU and
    // as of wgpu 22+, 256 is the value of 'min_storage_buffer_offset_alignment' for metal GPUs.
    let handle1 = handle1
        .offset_start(64 * core::mem::size_of::<u32>() as u64)
        .offset_end(64 * core::mem::size_of::<u32>() as u64);

    unsafe {
        kernel_buffer_len::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            TensorArg::from_raw_parts::<u32>(&handle1, &[32, 16, 4, 1], &[4, 4, 4, 8], 2),
        )
    };

    let actual = client.read_one(handle1.binding());
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual[0], 64);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_metadata {
    () => {
        use super::*;

        #[test]
        fn test_shape() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_shape_dim_4::<TestRuntime>(client.clone());
            cubecl_core::runtime_tests::metadata::test_shape_different_ranks::<TestRuntime>(client);
        }

        #[test]
        fn test_stride() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_stride_different_ranks::<TestRuntime>(
                client,
            );
        }

        #[test]
        fn test_len() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_len_different_ranks::<TestRuntime>(client);
        }

        #[test]
        fn test_buffer_len_discontiguous() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_buffer_len_discontiguous::<TestRuntime>(
                client,
            );
        }

        #[test]
        fn test_buffer_len_vectorized() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_buffer_len_vectorized::<TestRuntime>(client);
        }

        #[test]
        fn test_buffer_len_offset() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::metadata::test_buffer_len_offset::<TestRuntime>(client);
        }
    };
}

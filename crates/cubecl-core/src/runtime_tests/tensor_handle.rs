use crate::prelude::*; // brings TensorArgError, TensorHandleError, TensorHandleRef, Runtime, ComputeClient

fn make_client<R: Runtime>() -> ComputeClient<R::Server, R::Channel> {
    R::client(&R::Device::default())
}

pub fn test_handle_try_from_typed_ok_and_vec_checked_ok<R: Runtime>() {
    let client = make_client::<R>();
    let shape = vec![2usize, 8usize];
    let strides = compact_strides(&shape);
    let bytes = bytemuck::cast_slice::<f32, u8>(&vec![0.0f32; shape.iter().product()]).to_vec();
    let handle = client.create(&bytes);

    let href = TensorHandleRef::<R>::try_from_typed::<f32>(&handle, &strides, &shape).expect("ok");

    // Pick a supported factor that divides last dim (if any), else 1
    let mut picked = 1u8;
    for f in R::supported_line_sizes() {
        let f8 = (*f) as u8;
        if f8 > 1 && shape[1] % (*f as usize) == 0 {
            picked = f8;
            break;
        }
    }
    let _arg = href.try_as_tensor_arg(picked).expect("vec ok");
}

pub fn test_handle_try_from_parts_rank_mismatch<R: Runtime>() {
    let client = make_client::<R>();
    let shape = vec![2usize, 4usize];
    let strides_good = compact_strides(&shape);
    let bytes = bytemuck::cast_slice::<f32, u8>(&vec![0.0f32; shape.iter().product()]).to_vec();
    let handle = client.create(&bytes);

    let err = TensorHandleRef::<R>::try_from_parts(
        &handle,
        &strides_good[..1],
        &shape,
        core::mem::size_of::<f32>(),
    )
    .unwrap_err();
    match err {
        TensorHandleError::RankMismatch { .. } => {}
        _ => panic!("wrong error: {err:?}"),
    }
}

pub fn test_handle_try_from_parts_zero_stride<R: Runtime>() {
    let client = make_client::<R>();
    let shape = vec![2usize, 4usize];
    let mut strides = compact_strides(&shape);
    strides[0] = 0; // invalid when dim > 1
    let bytes = bytemuck::cast_slice::<f32, u8>(&vec![0.0f32; shape.iter().product()]).to_vec();
    let handle = client.create(&bytes);

    let err = TensorHandleRef::<R>::try_from_parts(
        &handle,
        &strides,
        &shape,
        core::mem::size_of::<f32>(),
    )
    .unwrap_err();
    match err {
        TensorHandleError::ZeroStride { .. } => {}
        _ => panic!("wrong error: {err:?}"),
    }
}

pub fn test_vec_checked_unsupported_factor<R: Runtime>() {
    let client = make_client::<R>();
    let shape = vec![1usize, 8usize];
    let strides = compact_strides(&shape);
    let bytes = bytemuck::cast_slice::<f32, u8>(&vec![0.0f32; shape.iter().product()]).to_vec();
    let handle = client.create(&bytes);
    let href = TensorHandleRef::<R>::try_from_typed::<f32>(&handle, &strides, &shape).expect("ok");

    // pick factor 7 which is typically unsupported
    let err = href.try_as_tensor_arg(7).unwrap_err();
    match err {
        TensorArgError::UnsupportedVectorization { .. } => {}
        _ => panic!("wrong error: {err:?}"),
    }
}

pub fn test_vec_checked_noncontiguous_inner_allows_vectorized<R: Runtime>() {
    let client = make_client::<R>();
    let shape = vec![2usize, 8usize];
    let mut strides = compact_strides(&shape);
    // Make inner stride non-contiguous (allowed by checked API)
    strides[1] = 2;
    let bytes = bytemuck::cast_slice::<f32, u8>(&vec![0.0f32; shape.iter().product()]).to_vec();
    let handle = client.create(&bytes);

    let href = TensorHandleRef::<R>::try_from_parts(
        &handle,
        &strides,
        &shape,
        core::mem::size_of::<f32>(),
    )
    .expect("ok");

    // Choose a supported factor > 1 if available
    let mut picked = None;
    for f in R::supported_line_sizes() {
        if *f > 1 {
            picked = Some(*f as u8);
            break;
        }
    }
    if let Some(factor) = picked {
        let _ = href
            .try_as_tensor_arg(factor)
            .expect("non-contiguous inner allowed");
    }
}

// Misalignment (last dim not divisible by factor) is permitted; tail handling is kernel-specific.
// We do not error on that case in the checked API.

#[macro_export]
macro_rules! testgen_tensor_handle {
    () => {
        use super::*;

        #[test]
        fn test_tensor_handle_try_from_typed_ok_and_vec_checked_ok() {
            cubecl_core::runtime_tests::tensor_handle::test_handle_try_from_typed_ok_and_vec_checked_ok::<TestRuntime>();
        }

        #[test]
        fn test_tensor_handle_try_from_parts_rank_mismatch() {
            cubecl_core::runtime_tests::tensor_handle::test_handle_try_from_parts_rank_mismatch::<TestRuntime>();
        }

        #[test]
        fn test_tensor_handle_try_from_parts_zero_stride() {
            cubecl_core::runtime_tests::tensor_handle::test_handle_try_from_parts_zero_stride::<TestRuntime>();
        }

        #[test]
        fn test_vec_checked_unsupported_factor() {
            cubecl_core::runtime_tests::tensor_handle::test_vec_checked_unsupported_factor::<TestRuntime>();
        }

        #[test]
        fn test_vec_checked_noncontiguous_inner_allows_vectorized() {
            cubecl_core::runtime_tests::tensor_handle::test_vec_checked_noncontiguous_inner_allows_vectorized::<TestRuntime>();
        }

    };
}

fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    if rank == 0 {
        return vec![];
    }
    let mut strides = vec![0; rank];
    strides[rank - 1] = 1;
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

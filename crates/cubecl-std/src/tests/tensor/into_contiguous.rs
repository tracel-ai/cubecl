use cubecl_core::{CubeElement, prelude::Runtime};

use crate::tensor::{TensorHandle, into_contiguous_packed};

/// Contiguous, row-major strides for the given shape.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut current = 1usize;
    for d in (0..shape.len()).rev() {
        strides[d] = current;
        current *= shape[d];
    }
    strides
}

/// CPU reference that packs row-major `unpacked` values (each fitting in `bits` bits) along
/// `pack_dim`, producing the packed `u32` storage array. Consecutive values along `pack_dim`
/// occupy increasing bit slots of the same word, matching the kernel's packing convention.
fn pack_along(
    unpacked: &[u32],
    shape: &[usize],
    pack_dim: usize,
    packing: usize,
    bits: u32,
) -> Vec<u32> {
    let rank = shape.len();
    let mut storage_shape = shape.to_vec();
    storage_shape[pack_dim] = storage_shape[pack_dim].div_ceil(packing);
    let storage_strides = contiguous_strides(&storage_shape);
    let storage_len: usize = storage_shape.iter().product();
    let mask = (1u32 << bits) - 1;

    let mut out = vec![0u32; storage_len];
    for (q, &value) in unpacked.iter().enumerate() {
        let mut coords = vec![0usize; rank];
        let mut remainder = q;
        for d in (0..rank).rev() {
            coords[d] = remainder % shape[d];
            remainder /= shape[d];
        }
        let slot = (coords[pack_dim] % packing) as u32;
        let mut offset = 0usize;
        for d in 0..rank {
            let coord = if d == pack_dim {
                coords[d] / packing
            } else {
                coords[d]
            };
            offset += coord * storage_strides[d];
        }
        out[offset] |= (value & mask) << (slot * bits);
    }
    out
}

/// Repack `unpacked` (row-major over `shape`) from `in_pack_dim` to the innermost dim via
/// `into_contiguous_packed`, then assert the result matches the CPU reference. A layout change
/// must preserve the unpacked values, so a zeroed or under-written output fails here.
fn run_repack_case<R: Runtime>(
    device: &R::Device,
    shape: &[usize],
    in_pack_dim: usize,
    packing: usize,
    bits: u32,
) {
    let client = R::client(device);
    let dtype = u32::cube_type();
    let rank = shape.len();
    let num_unpacked: usize = shape.iter().product();

    // Deterministic non-zero payload so every packed word is non-zero.
    let unpacked: Vec<u32> = (0..num_unpacked).map(|q| ((q % 15) + 1) as u32).collect();

    // Input is packed along `in_pack_dim`.
    let input_storage = pack_along(&unpacked, shape, in_pack_dim, packing, bits);
    let mut in_storage_shape = shape.to_vec();
    in_storage_shape[in_pack_dim] = in_storage_shape[in_pack_dim].div_ceil(packing);
    // `packed_dim` is counted from the innermost dim.
    let packed_dim = rank - 1 - in_pack_dim;

    // `into_contiguous_packed` repacks onto the innermost dim.
    let expected = pack_along(&unpacked, shape, rank - 1, packing, bits);

    let input_handle = client.create_from_slice(u32::as_bytes(&input_storage));
    let input = TensorHandle::<R>::new_contiguous(in_storage_shape, input_handle, dtype);

    let output =
        into_contiguous_packed(&client, input.binding(), packed_dim, shape, packing, dtype);

    let bytes = client.read_one_unchecked_tensor(output.handle.clone().copy_descriptor(
        output.shape().clone(),
        output.strides().clone(),
        size_of::<u32>(),
    ));
    let actual = u32::from_bytes(&bytes);

    assert!(
        actual.iter().any(|&v| v != 0),
        "repacked output is all zeros: the write loop ran zero vectors per thread (shape {shape:?})",
    );
    assert_eq!(
        actual,
        &expected[..],
        "repacked PackedU32 tensor does not match the CPU reference (shape {shape:?}, in_pack_dim {in_pack_dim})",
    );
}

/// Original repro: output `vector_size` 2 is greater than `elems_per_unit` 1, which previously
/// truncated `vectors_per_thread` to zero and left the buffer all zeros.
pub fn test_into_contiguous_packed_repack<R: Runtime>(device: &R::Device) {
    run_repack_case::<R>(device, &[1, 8, 16], 1, 8, 4);
}

/// Output `vector_size` 1, the only case that was correct before the fix. Guards no-regression.
pub fn test_into_contiguous_packed_vector_size_one<R: Runtime>(device: &R::Device) {
    run_repack_case::<R>(device, &[1, 8, 8], 1, 8, 4);
}

/// Large enough that `elems_per_unit` exceeds 1 while the output vectorizes, so
/// `vectors_per_thread` is greater than 1. Exercises the `vector_size * elems_per_unit` factor
/// and the multi-vector write loop, which a unit-`elems_per_unit` case never reaches.
pub fn test_into_contiguous_packed_multi_vector<R: Runtime>(device: &R::Device) {
    run_repack_case::<R>(device, &[4096, 256], 0, 8, 4);
}

/// Large tensor whose output storage last dim is not a multiple of `vector_size * elems_per_unit`,
/// exercising the halving reduction of `num_elems_per_unit`.
pub fn test_into_contiguous_packed_halving<R: Runtime>(device: &R::Device) {
    run_repack_case::<R>(device, &[8192, 32], 0, 8, 4);
}

use cubecl_core::{CubeElement, prelude::Runtime};

use crate::tensor::{TensorHandle, copy_into, into_contiguous_packed};

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

pub fn test_into_contiguous_rank_mismatch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let dtype = f32::cube_type();

    // Layout produced by max_pool1d: NHWC storage viewed as NCHW [1, 2, 4, 1].
    let shape = vec![1usize, 2, 4, 1];
    let strides = vec![8usize, 1, 2, 2];
    let num_elems = 8;

    let data: Vec<f32> = (0..num_elems).map(|i| i as f32 + 1.0).collect();
    let input = TensorHandle::<R>::new(
        client.create_from_slice(f32::as_bytes(&data)),
        shape.clone(),
        strides.clone(),
        dtype,
    );
    // Same elements in linear order, but with the unit dim dropped.
    let out_shape = vec![1usize, 2, 4];
    let output = TensorHandle::<R>::new_contiguous(
        out_shape.clone(),
        client.empty(num_elems * size_of::<f32>()),
        dtype,
    );
    copy_into(&client, input.binding(), output.clone().binding(), dtype);

    let bytes = client.read_one_unchecked_tensor(output.handle.clone().copy_descriptor(
        output.shape().clone(),
        output.strides().clone(),
        size_of::<f32>(),
    ));
    let actual = f32::from_bytes(&bytes);

    let in_strides = contiguous_strides(&shape);
    let expected: Vec<f32> = (0..num_elems)
        .map(|q| {
            let src: usize = (0..shape.len())
                .map(|d| (q / in_strides[d]) % shape[d] * strides[d])
                .sum();
            data[src]
        })
        .collect();

    assert_eq!(
        actual,
        &expected[..],
        "rank-mismatched copy ({shape:?} strides {strides:?} -> {out_shape:?})",
    );
}

/// Copy a permuted (non-contiguous) view of a contiguous buffer into a contiguous output and
/// compare against a CPU reference. `perm[d]` is the source axis that becomes output axis `d`.
fn run_permuted_case<R: Runtime, E: CubeElement + From<u8> + PartialEq>(
    device: &R::Device,
    base_shape: &[usize],
    perm: &[usize],
) {
    let client = R::client(device);
    let dtype = E::cube_type();
    let base_strides = contiguous_strides(base_shape);
    let num_elems: usize = base_shape.iter().product();

    let shape: Vec<usize> = perm.iter().map(|&p| base_shape[p]).collect();
    let strides: Vec<usize> = perm.iter().map(|&p| base_strides[p]).collect();

    // Deterministic payload; every element is distinct modulo 251 so a misplaced write shows up.
    let data: Vec<E> = (0..num_elems)
        .map(|i| E::from((i % 251 + 1) as u8))
        .collect();

    let input = TensorHandle::<R>::new(
        client.create_from_slice(E::as_bytes(&data)),
        shape.clone(),
        strides.clone(),
        dtype,
    );
    let output = TensorHandle::<R>::new_contiguous(
        shape.clone(),
        client.empty(num_elems * size_of::<E>()),
        dtype,
    );
    copy_into(&client, input.binding(), output.clone().binding(), dtype);

    let bytes = client.read_one_unchecked_tensor(output.handle.clone().copy_descriptor(
        output.shape().clone(),
        output.strides().clone(),
        size_of::<E>(),
    ));
    let actual = E::from_bytes(&bytes);

    let out_strides = contiguous_strides(&shape);
    let expected: Vec<E> = (0..num_elems)
        .map(|q| {
            let src: usize = (0..shape.len())
                .map(|d| (q / out_strides[d]) % shape[d] * strides[d])
                .sum();
            data[src]
        })
        .collect();

    assert_eq!(
        actual,
        &expected[..],
        "permuted copy mismatch (base shape {base_shape:?}, perm {perm:?}, \
         view shape {shape:?}, view strides {strides:?})",
    );
}

/// Repro for a bool `swap_dims(0, 2)` in Burn: the perpendicular copy kernel gathers
/// `vector_size` consecutive elements along the input's unit-stride axis, whose shape (3) is
/// not a multiple of the vector size picked for `u8` (2).
pub fn test_into_contiguous_permuted_unaligned_axis<R: Runtime>(device: &R::Device) {
    run_permuted_case::<R, u8>(device, &[2, 2, 3], &[2, 1, 0]);
}

/// Sweep small shapes and every permutation, for a 1-byte and a 4-byte element type, so a
/// vector size is only used when it is actually valid for the layout.
pub fn test_into_contiguous_permuted_sweep<R: Runtime>(device: &R::Device) {
    let shapes: &[&[usize]] = &[
        &[2, 3],
        &[3, 2],
        &[4, 6],
        &[6, 4],
        &[2, 2, 3],
        &[3, 2, 2],
        &[2, 3, 4],
        &[4, 3, 2],
        &[8, 3, 4],
        &[3, 8, 5],
        &[16, 5],
        &[5, 16],
        &[8, 8],
        &[16, 16],
        &[32, 2, 3],
        &[2, 3, 4, 5],
    ];

    for shape in shapes {
        for perm in permutations(shape.len()) {
            run_permuted_case::<R, u8>(device, shape, &perm);
            run_permuted_case::<R, u32>(device, shape, &perm);
        }
    }
}

/// All permutations of `0..rank`.
fn permutations(rank: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(rank);
    fn recurse(rank: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if current.len() == rank {
            out.push(current.clone());
            return;
        }
        for axis in 0..rank {
            if !current.contains(&axis) {
                current.push(axis);
                recurse(rank, current, out);
                current.pop();
            }
        }
    }
    recurse(rank, &mut current, &mut out);
    out
}

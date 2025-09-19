#![cfg(feature = "std")]

use cubecl_common::{future::block_on, reader::read_sync};
use cubecl_runtime::server::CopyDescriptor;

#[test]
fn wgpu_strided_io_roundtrip_u8_rows_pitched() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    let rows: usize = 4;
    let cols: usize = 5;
    let pitch_elems: usize = 8; // >= cols; introduces padding per row
    let elem_size: usize = 1; // u8
    let total_bytes = rows * pitch_elems * elem_size;

    // Allocate a buffer large enough to hold the pitched rows.
    let handle = client.empty(total_bytes);

    // Prepare contiguous row-major data (no padding in source).
    let mut data = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = (r as u8) * 32 + (c as u8);
        }
    }

    let binding = handle.clone().binding();
    let shape = [rows, cols];
    let strides = [pitch_elems, 1];

    // Write using pitched descriptor; the runtime should place each row at row_pitch offsets.
    let write_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    block_on(client.write_async(vec![(write_desc, &data)])).expect("pitched write ok");

    // Read back using the same pitched descriptor. The runtime reconstructs contiguous rows.
    let read_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    let out = read_sync(client.read_tensor_async(vec![read_desc]));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], data);
}

#[test]
fn wgpu_strided_io_roundtrip_f32_rows_pitched() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    let rows: usize = 3;
    let cols: usize = 7;
    let pitch_elems: usize = 10; // >= cols; introduces padding per row
    let elem_size: usize = core::mem::size_of::<f32>();
    let total_bytes = rows * pitch_elems * elem_size;

    // Allocate a buffer large enough to hold the pitched rows.
    let handle = client.empty(total_bytes);

    // Prepare contiguous row-major data (no padding in source).
    let mut data = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = (r as f32) * 100.0 + (c as f32);
        }
    }
    let bytes: &[u8] = bytemuck::cast_slice(&data);

    let binding = handle.clone().binding();
    let shape = [rows, cols];
    let strides = [pitch_elems, 1];

    // Write using pitched descriptor; the runtime should place each row at row_pitch offsets.
    let write_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    block_on(client.write_async(vec![(write_desc, bytes)])).expect("pitched write ok");

    // Read back using the same pitched descriptor. The runtime reconstructs contiguous rows.
    let read_desc = CopyDescriptor::new(binding.clone(), &shape, &strides, elem_size);
    let out = read_sync(client.read_tensor_async(vec![read_desc]));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], bytes);
}

#[test]
#[should_panic]
fn wgpu_strided_io_read_unsupported_strides_panics_rank3() {
    type R = cubecl_wgpu::WgpuRuntime;
    let client = R::client(&R::Device::default());

    // Rank 3 descriptor with non-trivial strides is currently unsupported on WGPU.
    let shape = [2usize, 3usize, 4usize];
    let strides = [12usize, 4usize, 1usize];
    let elem_size = 1usize;
    let total_bytes = shape.iter().product::<usize>() * elem_size;

    let handle = client.empty(total_bytes);
    let binding = handle.binding();

    // Attempting to read should surface UnsupportedStrides, which panics via client read wrapper.
    let desc = CopyDescriptor::new(binding, &shape, &strides, elem_size);
    let _ = client.read_tensor(vec![desc]);
}

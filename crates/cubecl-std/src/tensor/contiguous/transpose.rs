//! Coalesced F32 copies when the input and output have different unit-stride axes.

use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, prelude::*};

use super::base::{compact_strides, is_contiguous};

const TILE: usize = 32;
const ROWS: usize = 8;

/// Try a shared-memory transpose, leaving unsupported layouts to the general
/// copy kernel. A contiguous output can also be viewed in the input's shape,
/// which covers reshape copies without assuming the two ranks are equal.
pub(super) fn try_copy_transposed(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    dtype: ElemType,
) -> bool {
    let hardware = &client.properties().hardware;
    let rank = input.shape.len();
    let elements = input.shape.iter().product::<usize>();
    if dtype != f32::elem_type_native()
        || hardware.num_cpu_cores.is_some()
        || hardware.max_units_per_cube < (TILE * ROWS) as u32
        || hardware.max_shared_memory_size < TILE * (TILE + 1) * size_of::<f32>()
        || rank < 2
        || elements < 4096
        || elements != output.shape.iter().product::<usize>()
        || input.tiling.is_tiled()
        || output.tiling.is_tiled()
    {
        return false;
    }
    let last = rank - 1;
    if input.strides[last] == 1 || input.shape[last] < 16 {
        return false;
    }
    let Some(axis) = (0..last).find(|&axis| input.strides[axis] == 1 && input.shape[axis] >= 16)
    else {
        return false;
    };
    let mut output = output.clone();
    if input.shape != output.shape {
        if !is_contiguous(&output.shape, &output.strides) {
            return false;
        }
        output.shape = input.shape.clone();
        output.strides = compact_strides(&input.shape);
    }
    if output.strides[last] != 1 {
        return false;
    }

    let tiles_a = input.shape[axis].div_ceil(TILE);
    let tiles_b = input.shape[last].div_ceil(TILE);
    let batches = elements / (input.shape[axis] * input.shape[last]);
    let cube_dim = CubeDim::new_2d(TILE as u32, ROWS as u32);
    let cube_count =
        calculate_cube_count_elemwise(client, batches * tiles_a * tiles_b * TILE * ROWS, cube_dim);
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));
    copy_transposed::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        input.clone().into_tensor_arg(),
        output.into_tensor_arg(),
        tiles_a,
        tiles_b,
        batches,
        axis,
        rank,
    );
    true
}

#[cube(launch, address_type = "dynamic")]
fn copy_transposed(
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    tiles_a: usize,
    tiles_b: usize,
    batches: usize,
    #[comptime] axis: usize,
    #[comptime] rank: usize,
) {
    let block = CUBE_POS as usize;
    let mut batch = block / (tiles_a * tiles_b);
    if batch >= batches {
        terminate!();
    }
    let a = (block % tiles_a) * TILE;
    let b = ((block / tiles_a) % tiles_b) * TILE;
    let mut input_base = 0usize;
    let mut output_base = 0usize;
    #[unroll]
    for rev in 0..rank {
        let dim = rank - 1 - rev;
        if dim != axis && dim != rank - 1 {
            let coordinate = batch % input.shape(dim);
            batch /= input.shape(dim);
            input_base += coordinate * input.stride(dim);
            output_base += coordinate * output.stride(dim);
        }
    }

    // Padding the shared rows avoids bank conflicts when reading columns.
    let mut tile = Shared::<[f32]>::new_slice(TILE * (TILE + 1));
    let x = UNIT_POS_X as usize;
    let y = UNIT_POS_Y as usize;
    #[unroll]
    for step in 0..TILE / ROWS {
        let row = y + step * ROWS;
        if a + x < input.shape(axis) && b + row < input.shape(rank - 1) {
            let offset =
                input_base + (a + x) * input.stride(axis) + (b + row) * input.stride(rank - 1);
            tile[row * (TILE + 1) + x] = input[offset];
        }
    }
    sync_cube();
    #[unroll]
    for step in 0..TILE / ROWS {
        let row = y + step * ROWS;
        if a + row < input.shape(axis) && b + x < input.shape(rank - 1) {
            let offset =
                output_base + (a + row) * output.stride(axis) + (b + x) * output.stride(rank - 1);
            output[offset] = tile[x * (TILE + 1) + row];
        }
    }
}

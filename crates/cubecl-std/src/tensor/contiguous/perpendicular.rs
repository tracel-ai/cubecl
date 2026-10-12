use cubecl::prelude::*;
use cubecl_core::{
    self as cubecl, calculate_cube_count_elemwise, ir::VectorSize, tensor_vector_size_parallel,
    tensor_vector_size_perpendicular,
};
use std::cmp::min;

/// Copies a tensor whose unit-stride axis is not the last one into an output whose last axis
/// is, by transposing `vector_size × vector_size` blocks in registers.
///
/// For each block, a unit gathers `vector_size` vectors along the input's unit-stride axis,
/// transposes them, and stores `vector_size` vectors along the output's last axis. A unit
/// handles `blocks_per_unit` blocks in a row along the last axis.
///
/// A row is one block's worth of `axis` in one batch, where every axis but `axis` and the last
/// is a batch. Units walk the last axis fastest, so stores are coalesced across a cube, while
/// the loads of neighbouring rows fill the same cache lines.
#[cube(launch_unchecked, address_type = "dynamic")]
fn copy_perpendicular<T: Numeric, N: Size>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    blocks_last: usize,
    blocks_axis: usize,
    rows: usize,
    blocks_per_unit: usize,
    tiles_last: usize,
    panel_rows: usize,
    num_tiles: usize,
    #[comptime] axis: usize,
    #[comptime] rank: usize,
    #[define(T)] _elem: ElemType,
) {
    let tile = CUBE_POS;
    if tile >= num_tiles {
        terminate!()
    }

    let vector_size = input.vector_size();
    let last_axis = rank - 1;

    // Tiles are walked a panel of `panel_rows` row tiles at a time, about one batch, so cubes
    // running side by side read the same cache lines. Within a panel they are walked
    // diagonally: with power-of-two strides, cubes on the same input rows would otherwise all
    // hit the same memory channels.
    let panel_tiles = panel_rows * tiles_last;
    let tile_in_panel = tile % panel_tiles;
    let tile_rows_in_panel = tile_in_panel % panel_rows;
    let tile_rows = (tile / panel_tiles) * panel_rows + tile_rows_in_panel;
    let tile_last = (tile_in_panel / panel_rows + tile_rows_in_panel) % tiles_last;
    let row = tile_rows * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
    let block_start = (tile_last * CUBE_DIM_X as usize + UNIT_POS_X as usize) * blocks_per_unit;
    if row >= rows || block_start >= blocks_last {
        terminate!()
    }
    let mut block_end = block_start + blocks_per_unit;
    if block_end > blocks_last {
        block_end = blocks_last;
    }

    let block_axis = row % blocks_axis;
    let mut batch = row / blocks_axis;
    let mut offset_input = 0;
    let mut offset_output = 0;
    #[unroll]
    for i in 0..rank {
        let dim = rank - 1 - i;
        if dim != axis && dim != last_axis {
            let coordinate = batch % output.shape(dim);
            batch /= output.shape(dim);
            offset_input += coordinate * input.stride(dim);
            offset_output += coordinate * output.stride(dim);
        }
    }

    let start_axis = block_axis * vector_size;
    let input_stride = input.stride(last_axis);
    let output_stride = output.stride(axis);
    let index_input = (offset_input + start_axis) / vector_size;
    let index_output = (offset_output + start_axis * output_stride) / vector_size;
    let input_stride = input_stride / vector_size;
    let output_stride = output_stride / vector_size;

    let mut accumulators = Sequence::<Vector<T, N>>::new();

    #[unroll]
    for _ in 0..vector_size {
        accumulators.push(Vector::empty());
    }

    for block in block_start..block_end {
        // A block spans `vector_size` positions of the last axis: as many input strides, and a
        // single output vector.
        let index_input = index_input + block * vector_size * input_stride;
        let index_output = index_output + block;

        #[unroll]
        for i in 0..vector_size {
            let gathered = input[index_input + i * input_stride];

            #[unroll]
            for o in 0..vector_size {
                let vector = accumulators.index_mut(o);
                vector.insert(i, gathered.extract(o));
            }
        }

        #[unroll]
        for o in 0..vector_size {
            output[index_output + o * output_stride] = accumulators[o];
        }
    }
}

/// Copies `input` into `output` with the perpendicular kernel when its blocks can be
/// vectorized, returning whether it did.
///
/// Without vectorization a block is a single element and the kernel is a strided copy, so the
/// caller keeps its general path for that case. `input` and `output` must have the same shape.
pub fn try_launch_copy_perpendicular(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    dtype: ElemType,
) -> bool {
    let (axis, vector_size) = perpendicular_vector_size(client, input, output, dtype);
    if vector_size == 1 {
        return false;
    }

    let rank = output.shape.len();
    let blocks_last = output.shape[rank - 1] / vector_size;
    let blocks_axis = output.shape[axis] / vector_size;
    let batches =
        output.shape.iter().product::<usize>() / (output.shape[rank - 1] * output.shape[axis]);
    let rows = batches * blocks_axis;

    let (blocks_per_unit, cube_dim) = if client.properties().hardware.num_cpu_cores.is_some() {
        // Cubes run one after another on a CPU, so a cube spreads rows over the threads, and
        // each unit walks its whole row: a regular stride to prefetch, and one setup per row.
        let cube_dim = CubeDim::new(client, rows);
        (blocks_last, CubeDim::new_2d(1, cube_dim.num_elems()))
    } else {
        // A cube covers up to 8 blocks along the last axis and fills the rest of 256 units with
        // rows, so its loads share cache lines while its stores stay coalesced. A side shorter
        // than the cube shrinks it, and the other side takes the units it frees.
        let blocks_last_pow2 = blocks_last.next_power_of_two();
        let cube_x = blocks_last_pow2.min(8);
        let cube_y = rows.next_power_of_two().min(256 / cube_x);
        let cube_x = blocks_last_pow2.min(256 / cube_y);
        (1, CubeDim::new_2d(cube_x as u32, cube_y as u32))
    };

    let tiles_last = blocks_last
        .div_ceil(blocks_per_unit)
        .div_ceil(cube_dim.x as usize);
    let tiles_rows = rows.div_ceil(cube_dim.y as usize);
    let panel_rows = blocks_axis.div_ceil(cube_dim.y as usize);
    // The last panel may be short; its tiles past `rows` exit at once.
    let num_tiles = tiles_rows.div_ceil(panel_rows) * panel_rows * tiles_last;
    let cube_count = calculate_cube_count_elemwise(client, num_tiles, CubeDim::new_1d(1));
    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        copy_perpendicular::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            blocks_last,
            blocks_axis,
            rows,
            blocks_per_unit,
            tiles_last,
            panel_rows,
            num_tiles,
            axis,
            rank,
            dtype,
        );
    }

    true
}

/// The input's unit-stride axis, and the vector size the perpendicular kernel can use.
fn perpendicular_vector_size(
    client: &Client,
    input: &TensorBinding,
    output: &TensorBinding,
    dtype: ElemType,
) -> (usize, VectorSize) {
    let axis = input
        .strides
        .iter()
        .position(|stride| *stride == 1)
        .unwrap_or(0);
    let rank = output.shape.len();

    let vector_size_perpendicular = tensor_vector_size_perpendicular(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        rank - 1,
    );
    let vector_size_parallel = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &output.shape,
        &output.strides,
        rank - 1,
    );
    // The gather loads `vector_size` consecutive elements along the input's unit-stride axis,
    // so that axis must be vectorizable on its own.
    let vector_size_axis = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        axis,
    );
    let vector_size = min(
        min(vector_size_perpendicular, vector_size_parallel),
        vector_size_axis,
    );

    (axis, vector_size)
}

use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

pub fn shared_sum<R: Runtime, N: Numeric + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    cube_count: u32,
) -> Result<N, MissingAtomicAdd> {
    let atomic_elem = Atomic::<N>::as_elem_native_unchecked();
    if !client
        .properties()
        .feature_enabled(cubecl_core::Feature::Type(atomic_elem))
        || !client
            .properties()
            .feature_enabled(cubecl_core::Feature::AtomicFloat(
                cubecl_core::AtomicFeature::Add,
            ))
    {
        return Err(MissingAtomicAdd(atomic_elem));
    }

    let input_len = input.shape.iter().map(|s| *s as u32).product::<u32>();

    let elem = N::as_elem_native_unchecked();
    let line_size = R::line_size_elem(&elem)
        .filter(|line_size| input_len % *line_size as u32 == 0)
        .max()
        .unwrap_or(1) as u32;

    let cube_dim = CubeDim::new_2d(32, 8);

    let num_units = cube_count * cube_dim.num_elems();
    let num_lines_per_unit = input_len.div_ceil(num_units * line_size);

    let cube_count = CubeCount::new_1d(cube_count);

    let output_handle = client.create(N::as_bytes(&[N::from_int(0)]));
    let output_shape = vec![1];
    let output_stride = vec![1];

    let output = unsafe {
        TensorHandleRef::<R>::from_raw_parts(
            &output_handle,
            &output_stride,
            &output_shape,
            size_of::<N>(),
        )
    };

    unsafe {
        shared_sum_kernel::launch_unchecked::<N, R>(
            client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(line_size as u8),
            output.as_tensor_arg(1),
            cube_dim.num_elems(),
            line_size,
            num_lines_per_unit,
        );
    }

    let binding = output_handle.binding();
    let bytes = client.read_one(binding);
    let output_values = N::from_bytes(&bytes);

    Ok(output_values[0])
}

#[cube(launch_unchecked)]
fn shared_sum_kernel<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Atomic<N>>,
    #[comptime] shared_memory_size: u32,
    #[comptime] line_size: u32,
    #[comptime] num_lines_per_unit: u32,
) {
    let mut shared_memory = SharedMemory::new_lined(shared_memory_size, line_size);
    shared_memory[UNIT_POS] = Line::empty(line_size).fill(N::from_int(0));

    let start = ABSOLUTE_POS * num_lines_per_unit;
    let end = start + num_lines_per_unit;

    // Prevent out-of-bound access
    let start = select(start < input.len(), start, input.len());
    let end = select(end < input.len(), end, input.len());

    for k in start..end {
        shared_memory[UNIT_POS] += input[k];
    }

    let line = sum_shared_memory(&mut shared_memory);

    let mut sum = N::from_int(0);
    #[unroll]
    for k in 0..line_size {
        sum[k] += line[k];
    }
    if UNIT_POS == 0 {
        Atomic::add(&output[0], sum);
    }
}

#[cube]
fn sum_shared_memory<N: Numeric>(accumulator: &mut SharedMemory<Line<N>>) -> Line<N> {
    sync_units();
    let mut num_active_units = CUBE_DIM;
    let mut jump = 1;
    while num_active_units > 1 {
        num_active_units /= 2;
        let destination = jump * 2 * UNIT_POS;
        let origin = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active_units {
            let element = accumulator[origin];
            accumulator[destination] += element;
        }
        jump *= 2;
        sync_units();
    }
    accumulator[0]
}

#[derive(Debug)]
pub struct MissingAtomicAdd(Elem);

impl std::fmt::Display for MissingAtomicAdd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elem = self.0;
        write!(f, "Atomic add not supported by the client for {elem}")
    }
}

impl std::error::Error for MissingAtomicAdd {}

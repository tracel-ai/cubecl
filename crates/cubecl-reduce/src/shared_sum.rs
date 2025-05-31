use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::ReduceError;

/// Sum all the elements of the input tensor distributed over `cube_count` cubes.
///
/// This is an optimized version for summing large tensors using multiple cubes.
/// For summing a single axis, the regular [reduce] entry point is preferred.
///
/// Return an error if atomic addition is not supported for the type `N`.
///
/// # Important
///
/// This doesn't set the value of output to 0 before computing the sums.
/// It is the responsibility of the caller to ensure that output is set to
/// the proper value. Basically, the behavior of this kernel is akin to the AddAssign operator
/// as it update the output instead of overwriting it.
///
/// # Example
///
/// This examples show how to sum all the elements of a small `2 x 2` matrix.
/// For more details, see the CubeCL documentation.
///
/// ```ignore
/// let client = /* ... */;
/// let size_f32 = std::mem::size_of::<f32>();
///
/// // Create input and output handles.
/// let input_handle = client.create(f32::as_bytes(&[0, 1, 2, 3]));
/// let output_handle = client.empty(size_of::<F>());
/// let input = unsafe {
///     TensorHandleRef::<R>::from_raw_parts(
///         &input_handle,
///         &[2, 1],
///         &[2, 2],
///         size_f32,
///     )
/// };
/// let output = unsafe {
///     TensorHandleRef::<R>::from_raw_parts(&output_handle, &[1], &[1], size_of::<F>())
/// };
///
/// // Here `R` is a `cubecl::Runtime`.
/// let result = shared_sum::<R, f32>(&client, input, output, cube_count);
///
/// if result.is_ok() {
///        let binding = output_handle.binding();
///        let bytes = client.read_one(binding);
///        let output_values = f32::from_bytes(&bytes);
///        println!("Output = {:?}", output_values); // Should print [6].
/// }
/// ```
pub fn shared_sum<R: Runtime, N: Numeric + CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    cube_count: u32,
) -> Result<(), ReduceError> {
    // Check that the client supports atomic addition.
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
        return Err(ReduceError::MissingAtomicAdd(N::as_elem_native_unchecked()));
    }

    let input_len = input.shape.iter().map(|s| *s as u32).product::<u32>();

    // Compute the optimal line size.
    let elem = N::as_elem_native_unchecked();
    let line_size = R::line_size_elem(&elem)
        .filter(|line_size| input_len % *line_size as u32 == 0)
        .max()
        .unwrap_or(1) as u32;

    // Compute extra parameters.
    let cube_dim = CubeDim::new_2d(32, 8); // NOTE: If you change that, keep the unit count a power of 2.
    let num_units = cube_count * cube_dim.num_elems();
    let num_lines_per_unit = input_len.div_ceil(num_units * line_size);
    let cube_count = CubeCount::new_1d(cube_count);

    // Launch kernel
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

    Ok(())
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

    // Each unit reduce `num_lines_per_unit` lines.
    let start = ABSOLUTE_POS * num_lines_per_unit;
    let end = start + num_lines_per_unit;

    // Prevent out-of-bound access
    let start = select(start < input.len(), start, input.len());
    let end = select(end < input.len(), end, input.len());

    // Each unit sum its lines.
    for k in start..end {
        shared_memory[UNIT_POS] += input[k];
    }

    // Sum all lines within the shared_memory to a single line.
    let line = sum_shared_memory(&mut shared_memory);

    // Sum all the elements within the line.
    let sum = RuntimeCell::<N>::new(N::from_int(0));
    #[unroll]
    for k in 0..line_size {
        let update = line[k] + sum.read();
        sum.store(update);
    }

    // Add the sum for the current cube to the output.
    if UNIT_POS == 0 {
        Atomic::add(&output[0], sum.consume());
    }
}

// This is a simplified version of [tree_reduce].
// See the documentation there for details.
// Here we assume that `CUBE_DIM` is always a power of two.
#[cube]
fn sum_shared_memory<N: Numeric>(accumulator: &mut SharedMemory<Line<N>>) -> Line<N> {
    sync_cube();
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
        sync_cube();
    }
    accumulator[0]
}

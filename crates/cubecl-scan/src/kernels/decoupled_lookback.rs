use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::TensorHandle;

use crate::{ScanError, instructions::ScanInstruction};

// NOTE: this is only a simple scan for now (no support for tensor dimensions)

type Flag = u32;

const FLAG_AGGREGATE_AVAILABLE: Flag = 1;
const FLAG_PREFIX_AVAILABLE: Flag = 2;

#[cube]
fn to_u32<N: Numeric>(val: N, #[comptime] size: u32) -> u32 {
    match size {
        1 => u32::cast_from(u8::reinterpret(val)),
        2 => u32::cast_from(u16::reinterpret(val)),
        4 => u32::reinterpret(val),
        _ => panic!("Unsupported size {size}"),
    }
}

#[cube]
fn from_u32<N: Numeric>(val: u32, #[comptime] size: u32) -> N {
    match size {
        1 => N::reinterpret(u8::cast_from(val)),
        2 => N::reinterpret(u16::cast_from(val)),
        4 => N::reinterpret(val),
        _ => panic!("Unsupported size {size}"),
    }
}

#[cube(launch, launch_unchecked)]
pub fn decoupled_lookback_scan_kernel<N: Numeric, I: ScanInstruction>(
    input: &Array<Line<N>>,
    aggregates: &mut Array<Atomic<u32>>,
    flags: &mut Array<Atomic<Flag>>,
    output: &mut Array<Line<N>>,
    #[comptime] line_size: u32,
    #[comptime] inclusive: bool,
    #[comptime] elem_size: u32,
) {
    let partition_idx = CUBE_POS;
    let val = select(
        ABSOLUTE_POS < input.len(),
        input[ABSOLUTE_POS],
        Line::<N>::empty(line_size),
    );

    let local_aggregate = I::aggregate_line::<N>(val, line_size);
    let plane_scan = I::scan_plane::<N>(local_aggregate, false);
    let aggregate = plane_scan + local_aggregate;

    // Perform the aggregate broadcasting step
    let aggregate_idx = partition_idx * 2;
    if ABSOLUTE_POS == PLANE_DIM - 1 {
        // Handle the first partition
        // Aggregate
        Atomic::store(
            &aggregates[aggregate_idx],
            to_u32::<N>(aggregate, elem_size),
        );
        // Prefix
        Atomic::store(
            &aggregates[aggregate_idx + 1],
            to_u32::<N>(aggregate, elem_size),
        );

        sync_storage();
        // Mark the prefix as available
        Atomic::store(&flags[partition_idx], FLAG_PREFIX_AVAILABLE);
    } else if UNIT_POS == PLANE_DIM - 1 {
        // Handle all other partitions
        Atomic::store(
            &aggregates[aggregate_idx],
            to_u32::<N>(aggregate, elem_size),
        );

        sync_storage();
        // Mark the aggregate as available
        Atomic::store(&flags[partition_idx], FLAG_AGGREGATE_AVAILABLE);
    }
    sync_cube();
    sync_storage();

    let mut lookback_idx = partition_idx;
    let mut done: u32 = 0;
    let mut lookback_value = N::cast_from(0);
    while lookback_idx > 0 && done == 0 {
        if UNIT_POS == 0 {
            let desc_idx = lookback_idx - 1;
            let pred_flag = Atomic::load(&flags[desc_idx]);

            if pred_flag == FLAG_AGGREGATE_AVAILABLE {
                let aggregate = Atomic::load(&aggregates[desc_idx * 2]);
                lookback_value = I::apply::<N>(lookback_value, from_u32::<N>(aggregate, elem_size));
                lookback_idx -= 1;
            } else if pred_flag == FLAG_PREFIX_AVAILABLE {
                let aggregate = Atomic::load(&aggregates[desc_idx * 2 + 1]);
                lookback_value = I::apply::<N>(lookback_value, from_u32::<N>(aggregate, elem_size));
                done = 1;
            }
        }
        sync_cube();

        // Broadcast the "done" state to all threads in the plane
        done = plane_broadcast(done, 0);
    }
    sync_cube();
    // Fetch the computed lookback value into all threads in the plane
    lookback_value = plane_broadcast(lookback_value, 0);

    let scan_carry = lookback_value + aggregate;

    // Mark the prefix as available
    if UNIT_POS == PLANE_DIM - 1 {
        // Prefix
        Atomic::store(
            &aggregates[aggregate_idx + 1],
            to_u32::<N>(scan_carry, elem_size),
        );
        Atomic::store(&flags[partition_idx], FLAG_PREFIX_AVAILABLE);
    }
    sync_cube();
    sync_storage();

    let scan_res = I::scan_line::<N>(lookback_value + plane_scan, val, line_size, inclusive);
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = scan_res;
    }
}

pub fn launch_ref<R: Runtime, N: Numeric, I: ScanInstruction>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
    axis: usize,
    inclusive: bool,
) -> Result<(), ScanError> {
    let input = TensorHandle::<R, N>::from_ref(input);
    let output = TensorHandle::<R, N>::from_ref(output);

    launch::<R, N, I>(client, input, output, axis, inclusive)
}

pub fn launch<R: Runtime, N: Numeric, I: ScanInstruction>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: TensorHandle<R, N>,
    output: TensorHandle<R, N>,
    axis: usize,
    inclusive: bool,
) -> Result<(), ScanError> {
    use cubecl_core::Feature;
    let atomic_elem = Atomic::<Flag>::as_type_native_unchecked();
    let has_feature = |f| client.properties().feature_enabled(f);

    // Check that the client supports the provided type
    if !has_feature(Feature::Type(N::as_type_native_unchecked())) {
        return Err(ScanError::UnsupportedType(N::as_type_native_unchecked()));
    }

    // Check that the client supports atomic load/store
    if !has_feature(Feature::Type(atomic_elem))
        || !has_feature(Feature::AtomicUInt(cubecl_core::AtomicFeature::LoadStore))
    {
        return Err(ScanError::MissingAtomicLoadStore(
            Flag::as_type_native_unchecked(),
        ));
    }

    let num_elements = tensor_size(&input);
    if num_elements != tensor_size(&output) {
        return Err(ScanError::MismatchSize {
            shape_a: input.shape.clone(),
            shape_b: output.shape.clone(),
        });
    }
    if !precise_plane_dim::<R>(client) {
        return Err(ScanError::ImprecisePlaneDim);
    }

    let hw_props = &client.properties().hardware;
    let plane_size = hw_props.plane_size_max;

    // ToDo: do better line size selection for non-1 strides
    let line_size = match input.strides[axis] {
        1 => {
            let elem = N::as_type_native_unchecked();
            R::line_size_type(&elem)
                .filter(|s| num_elements % (*s as usize) == 0)
                .max()
                .unwrap_or(1) as u32
        }
        _ => 1,
    };

    let cube_dim = CubeDim::new_1d(plane_size);

    let block_elements = (line_size * plane_size) as usize;
    let num_blocks = num_elements.div_ceil(block_elements);

    // Round to the granularity used by the reset kernel
    let flags_per_cube = {
        let elem = Flag::as_type_native_unchecked();
        let line_size = R::line_size_type(&elem).max().unwrap_or(0) as usize;
        line_size * (plane_size as usize)
    };
    let num_flags = num_blocks.next_multiple_of(flags_per_cube);

    // Overwritten before any reads, so it can contain garbage initially
    let aggregates = client.empty(num_blocks * 2 * (u32::elem_size() as usize));
    let flags = TensorHandle::<R, Flag>::zeros(client, vec![num_flags]);

    dbg!(input.shape);
    dbg!(output.shape);
    dbg!(num_elements);
    dbg!(line_size);
    dbg!(plane_size);
    dbg!(block_elements);
    dbg!(num_blocks);
    dbg!(num_flags);

    unsafe {
        decoupled_lookback_scan_kernel::launch::<N, I, R>(
            client,
            CubeCount::Static(num_blocks as u32, 1, 1),
            cube_dim,
            ArrayArg::from_raw_parts::<N>(&input.handle, num_elements, line_size as u8),
            ArrayArg::from_raw_parts::<N>(&aggregates, num_blocks * 2, 1),
            ArrayArg::from_raw_parts::<Flag>(&flags.handle, num_flags, 1),
            ArrayArg::from_raw_parts::<N>(&output.handle, num_elements, line_size as u8),
            line_size,
            inclusive,
            N::elem_size(),
        );
    }

    Ok(())
}

fn tensor_size<R: Runtime, E: CubePrimitive>(handle: &TensorHandle<R, E>) -> usize {
    handle.shape.iter().product::<usize>()
}

fn precise_plane_dim<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> bool {
    let hw_props = &client.properties().hardware;
    hw_props.plane_size_min == hw_props.plane_size_max
}

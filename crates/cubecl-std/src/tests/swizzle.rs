use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::runtime::Runtime;

use crate::Swizzle;

const STAGE_ELEMENTS: usize = 1024;
const SWIZZLE_ATOM_BYTES: usize = 16;

#[cube(launch_unchecked)]
fn read_stage_swizzled_over_128_byte_spans<F: Float, N: Size>(
    stage_as_laid_out: &[Vector<F, N>],
    stage_in_logical_order: &mut [Vector<F, N>],
    #[comptime] stage_vectors: usize,
    #[comptime] vector_bytes: usize,
) {
    let mut stage = Shared::new_slice(stage_vectors);
    let stage_len = stage_as_laid_out.len();
    let steps = stage_len.div_ceil(CUBE_DIM as usize);

    for step in 0..steps {
        let index = UNIT_POS as usize + step * CUBE_DIM as usize;
        if index < stage_len {
            stage[index] = stage_as_laid_out[index];
        }
    }
    sync_cube();

    let swizzle = Swizzle::new(3u32, 4u32, 3);
    for step in 0..steps {
        let index = UNIT_POS as usize + step * CUBE_DIM as usize;
        if index < stage_len {
            let physical = swizzle.apply(index as u32, vector_bytes) as usize;
            stage_in_logical_order[index] = stage[physical];
        }
    }
}

/// A `load_width`-bit vector reads a stage swizzled over 128-byte spans back in logical order.
pub fn test_load_width_vectors_read_a_128_byte_swizzled_stage_in_logical_order<R: Runtime>(
    client: Client,
) {
    let hardware = &client.properties().hardware;
    // A CPU reports its register width, which outgrows a swizzle atom, and has no banked shared
    // memory for a swizzle to spread.
    if hardware.num_cpu_cores.is_some() {
        return;
    }

    let element_bits = 8 * size_of::<f32>();
    let vector_size = (hardware.load_width as usize / element_bits)
        .min(hardware.max_vector_size)
        .max(1);
    let stage_vectors = STAGE_ELEMENTS / vector_size;

    let logical: Vec<f32> = (0..STAGE_ELEMENTS).map(|index| index as f32).collect();
    let laid_out = lay_out_as_swizzled_over_128_byte_spans(&logical);

    let input = client.create_from_slice(f32::as_bytes(&laid_out));
    let output = client.empty(STAGE_ELEMENTS * size_of::<f32>());

    unsafe {
        read_stage_swizzled_over_128_byte_spans::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(&client, stage_vectors),
            vector_size,
            BufferArg::from_raw_parts(input, STAGE_ELEMENTS),
            BufferArg::from_raw_parts(output.clone(), STAGE_ELEMENTS),
            stage_vectors,
            vector_size * size_of::<f32>(),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = f32::from_bytes(&actual);
    assert_eq!(
        actual,
        &logical[..],
        "f32x{vector_size} vectors read the swizzled stage out of logical order"
    );
}

fn lay_out_as_swizzled_over_128_byte_spans(logical: &[f32]) -> Vec<f32> {
    let floats_per_atom = SWIZZLE_ATOM_BYTES / size_of::<f32>();
    let mut laid_out = vec![0.0; logical.len()];

    for atom in 0..logical.len() / floats_per_atom {
        let byte_offset = atom * SWIZZLE_ATOM_BYTES;
        let swizzled_byte_offset = byte_offset ^ ((byte_offset & (0b111 << 7)) >> 3);
        let swizzled_atom = swizzled_byte_offset / SWIZZLE_ATOM_BYTES;
        laid_out[swizzled_atom * floats_per_atom..][..floats_per_atom]
            .copy_from_slice(&logical[atom * floats_per_atom..][..floats_per_atom]);
    }

    laid_out
}

#[macro_export]
macro_rules! testgen_swizzle {
    () => {
        mod swizzle {
            use super::*;
            use $crate::tests::swizzle::*;

            #[$crate::tests::test_log::test]
            fn load_width_vectors_read_a_128_byte_swizzled_stage_in_logical_order() {
                let client = TestRuntime::client(&Default::default());
                test_load_width_vectors_read_a_128_byte_swizzled_stage_in_logical_order::<
                    TestRuntime,
                >(client);
            }
        }
    };
}

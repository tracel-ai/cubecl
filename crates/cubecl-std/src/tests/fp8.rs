use cubecl_common::quant::scheme::{QuantScheme, QuantStore, QuantValue};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::quant::unpack_cast_u32;

#[cube(launch_unchecked)]
fn kernel_unpack<NQ: Size, NF: Size>(
    words: &[Vector<u32, NQ>],
    out: &mut [Vector<f32, NF>],
    #[comptime] scheme: QuantScheme,
) {
    out[0] = unpack_cast_u32::<f32, NQ, NF>(words[0], ComptimeOption::new_None(), scheme);
}

/// Quantized fp8 reads decode in software: WGSL has no scalar fp8 to cast through.
pub fn test_fp8_unpack_decodes_in_software<R: Runtime>(client: Client) {
    let cases: [(QuantValue, u32, [f32; 4]); 2] = [
        (QuantValue::E4M3, 0x7EB8_3800, [0.0, 1.0, -1.0, 448.0]),
        (
            QuantValue::E5M2,
            0x017B_BC3C,
            [1.0, -1.0, 57344.0, 2f32.powi(-16)],
        ),
    ];
    for (value, word, expected) in cases {
        let scheme = QuantScheme::default()
            .with_value(value)
            .with_store(QuantStore::PackedU32(0));
        let input = client.create_from_slice(u32::as_bytes(&[word]));
        let out = client.empty(4 * size_of::<f32>());

        unsafe {
            kernel_unpack::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                1,
                4,
                BufferArg::from_raw_parts(input, 1),
                BufferArg::from_raw_parts(out.clone(), 4),
                scheme,
            )
        };

        let actual = f32::from_bytes(&client.read_one_unchecked(out)).to_vec();
        assert_eq!(actual, expected, "{value:?}");
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_fp8 {
    () => {
        mod fp8 {
            use super::*;

            #[$crate::tests::test_log::test]
            fn unpack_decodes_in_software() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::fp8::test_fp8_unpack_decodes_in_software::<TestRuntime>(client);
            }
        }
    };
}

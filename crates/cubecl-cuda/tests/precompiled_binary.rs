//! A kernel compiled outside `CubeCL` runs on `CubeCL`'s buffers and stream,
//! taking its parameters in its own order rather than `CubeCL`'s.

use cubecl_core::ir::{UIntKind, metadata::Info, settings::Dim3};
use cubecl_core::prelude::*;
use cubecl_core::{KernelParam, PrecompiledBinary};
use cubecl_cuda::CudaRuntime;
use cubecl_environment::bytes::Bytes;
use cubecl_server::kernel::{CubeKernel, KernelDefinition, KernelMetadata};
use cubecl_server::runtime::Runtime;
use cubecl_server::server::{KernelArguments, MetadataBindingInfo};

/// Scalars between the buffers, the way a cuTile kernel takes a tensor's
/// pointer followed by its shape: `CubeCL`'s own convention puts every buffer
/// first and packs the scalars after them.
const SOURCE: &str = r#"
extern "C" __global__ void axpy(float* out, int n, const float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + out[i];
    }
}
"#;

const BLOCK: u32 = 64;

/// The module image a foreign compiler would hand over. NVRTC stands in for
/// one here, and its PTX is a module the driver loads like a cubin. It is
/// handed over without a NUL, which the runtime adds.
fn image() -> Bytes {
    let ptx = cudarc::nvrtc::compile_ptx(SOURCE).expect("NVRTC compiles the test kernel");
    Bytes::from_bytes_vec(ptx.to_src().into_bytes())
}

struct Axpy;

impl KernelMetadata for Axpy {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(SOURCE)
            .cube_dim(Dim3::new_1d(BLOCK))
    }

    fn address_type(&self) -> ElemType {
        ElemType::UInt(UIntKind::U32)
    }
}

impl CubeKernel for Axpy {
    fn define(&self) -> KernelDefinition {
        let settings = KernelSettings::new(
            Dim3::new_1d(BLOCK),
            ExecutionMode::Checked,
            AddressType::U32,
        );
        KernelDefinition {
            body: Scope::root(settings.clone()),
            settings,
            info: Info::default(),
        }
    }

    fn binary(&self) -> Option<PrecompiledBinary> {
        Some(PrecompiledBinary {
            image: image(),
            entrypoint_name: "axpy".to_string(),
            params: vec![
                KernelParam::Resource(0),
                KernelParam::Info(0),
                KernelParam::Resource(1),
                KernelParam::Info(1),
            ],
        })
    }
}

#[test]
fn a_precompiled_binary_launches_with_its_own_parameter_order() {
    let client = CudaRuntime::client(&Default::default());
    // Not a multiple of the block, so the bound `n` has to arrive intact.
    let n = 100usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let out_init = vec![1.0f32; n];
    let a = 2.5f32;

    let x_handle = client.create_from_slice(f32::as_bytes(&x));
    let out_handle = client.create_from_slice(f32::as_bytes(&out_init));

    client.launch(
        Box::new(Axpy),
        CubeCount::Static((n as u32).div_ceil(BLOCK), 1, 1),
        KernelArguments::new()
            .with_buffer(out_handle.clone().binding())
            .with_buffer(x_handle.binding())
            .with_info(MetadataBindingInfo::custom(vec![
                n as u64,
                a.to_bits() as u64,
            ])),
    );

    let bytes = client.read_one(out_handle).expect("the launch ran");
    let output = f32::from_bytes(&bytes);
    let expected: Vec<f32> = x.iter().map(|x| a * x + 1.0).collect();
    assert_eq!(output, expected.as_slice());
}

#[test]
fn a_parameter_the_launch_does_not_have_is_refused() {
    let client = CudaRuntime::client(&Default::default());
    let out_handle = client.create_from_slice(f32::as_bytes(&[0.0f32; 4]));

    // One buffer and no info, where the kernel names two of each.
    client.launch(
        Box::new(Axpy),
        CubeCount::Static(1, 1, 1),
        KernelArguments::new().with_buffer(out_handle.clone().binding()),
    );

    assert!(
        client.read_one(out_handle).is_err(),
        "the output of a refused launch must not read back as written"
    );
}

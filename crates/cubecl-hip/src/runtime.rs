use cubecl_core::{
    ir::{Elem, FloatKind},
    Feature, FeatureSet, Properties, Runtime,
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::dynamic::{DynamicMemoryManagement, DynamicMemoryManagementOptions},
    ComputeRuntime,
};

use crate::{
    compiler::HipCompiler,
    compute::{HipContext, HipServer, HipStorage},
    device::HipDevice,
};

#[derive(Debug)]
pub struct HipRuntime;

static RUNTIME: ComputeRuntime<HipDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

const MEMORY_OFFSET_ALIGNMENT: u32 = 32;

type Server = HipServer<DynamicMemoryManagement<HipStorage>>;

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer<DynamicMemoryManagement<HipStorage>>;

    type Channel = MutexComputeChannel<HipServer<DynamicMemoryManagement<HipStorage>>>;
    type Device = HipDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        fn init(index: usize) -> HipContext<DynamicMemoryManagement<HipStorage>> {
            let mut ctx: cubecl_hip_sys::hipCtx_t = 0 as cubecl_hip_sys::hipCtx_t;
            unsafe {
                let status = cubecl_hip_sys::hipCtxCreate(&mut ctx, 0, index as cubecl_hip_sys::hipDevice_t);
                assert_eq!(status, HIP_SUCCESS, "Should create the HIP context");
            };

            let stream = unsafe {
                let mut stream: cubecl_hip_sys::hipStream_t = std::ptr::null_mut();
                let stream_status = cubecl_hip_sys::hipStreamCreate(&mut stream);
                assert_eq!(stream_status, HIP_SUCCESS, "Should create a stream");
                stream
            };

            let max_memory = unsafe {
                let free: usize = 0;
                let total: usize = 0;
                let status = cubecl_hip_sys::hipMemGetInfo(&free as *const _ as *mut usize, &total as *const _ as *mut usize);
                assert_eq!(status, HIP_SUCCESS, "Should get the available memory of the device");
                total
            };
            let storage = HipStorage::new(stream);
            let options = DynamicMemoryManagementOptions::preset(
                max_memory / 4, // Max chunk size is max_memory / 4
                MEMORY_OFFSET_ALIGNMENT as usize,
            );
            let memory_management = DynamicMemoryManagement::new(storage, options);
            HipContext::new(memory_management, stream, &mut ctx)
        }

        RUNTIME.client(device, move || {
            let server = HipServer::new(device.index, Box::new(init));
            let features = FeatureSet::new(&[Feature::Subcube]);

            // TODO
            // register_wmma_features(&mut features, server.arch_version());
            ComputeClient::new(
                MutexComputeChannel::new(server),
                features,
                Properties {
                    memory_offset_alignment: MEMORY_OFFSET_ALIGNMENT,
                },
            )
        })
    }

    fn name() -> &'static str {
        "cuda"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[8, 4, 2]
    }
}

fn register_wmma_features(features: &mut FeatureSet, arch: u32) {
    let wmma_minimum_version = 70;
    let mut wmma = false;

    if arch >= wmma_minimum_version {
        wmma = true;
    }

    if wmma {
        // Types fully supported.
        for (a, b, c) in [
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
            ),
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F32),
            ),
            (
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::F32),
            ),
        ] {
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 16,
                k: 16,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 32,
                k: 16,
                n: 8,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 8,
                k: 16,
                n: 32,
            });
        }
    }
}

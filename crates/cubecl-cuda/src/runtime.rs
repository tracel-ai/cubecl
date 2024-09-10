use std::mem::MaybeUninit;

use cubecl_core::{
    ir::{Elem, FloatKind},
    Feature, FeatureSet, Properties, Runtime,
};
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::dynamic::{DynamicMemoryManagement, DynamicMemoryManagementOptions},
    ComputeRuntime,
};

use crate::{
    compiler::CudaCompiler,
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};

#[derive(Debug)]
pub struct CudaRuntime;

static RUNTIME: ComputeRuntime<CudaDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

const MEMORY_OFFSET_ALIGNMENT: u32 = 32;

type Server = CudaServer<DynamicMemoryManagement<CudaStorage>>;

impl Runtime for CudaRuntime {
    type Compiler = CudaCompiler;
    type Server = CudaServer<DynamicMemoryManagement<CudaStorage>>;

    type Channel = MutexComputeChannel<CudaServer<DynamicMemoryManagement<CudaStorage>>>;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        fn init(index: usize) -> CudaContext<DynamicMemoryManagement<CudaStorage>> {
            cudarc::driver::result::init().unwrap();
            let device_ptr = cudarc::driver::result::device::get(index as i32).unwrap();
            let arch = unsafe {
                let major = cudarc::driver::result::device::get_attribute(device_ptr, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap();
                let minor = cudarc::driver::result::device::get_attribute(device_ptr, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap();
                major * 10 + minor
            } as u32;

            let ctx = unsafe {
                let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                cudarc::driver::result::ctx::set_current(ctx).unwrap();
                ctx
            };

            let stream = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .unwrap();
            let max_memory = unsafe {
                let mut bytes = MaybeUninit::uninit();
                cudarc::driver::sys::lib().cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_ptr);
                bytes.assume_init()
            };
            let storage = CudaStorage::new(stream);
            let options = DynamicMemoryManagementOptions::preset(
                max_memory / 4, // Max chunk size is max_memory / 4
                MEMORY_OFFSET_ALIGNMENT as usize,
            );
            let memory_management = DynamicMemoryManagement::new(storage, options);
            CudaContext::new(memory_management, stream, ctx, arch)
        }

        RUNTIME.client(device, move || {
            let mut server = CudaServer::new(device.index, Box::new(init));
            let mut features = FeatureSet::new(&[Feature::Subcube]);

            register_wmma_features(&mut features, server.arch_version());
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
                k: 8,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 8,
                k: 32,
                n: 16,
            });
        }
    }
}

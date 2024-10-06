use std::mem::MaybeUninit;

use cubecl_core::{
    ir::{Elem, FloatKind},
    Feature, MemoryConfiguration, Runtime,
};
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{dynamic::DynamicMemoryManagement, MemoryDeviceProperties},
    ClientProperties, ComputeRuntime,
};

use crate::{
    compiler::CudaCompiler,
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};

/// The values that control how a WGPU Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct CudaRuntime;

type Server = CudaServer<DynamicMemoryManagement<CudaStorage>>;
type Channel = MutexComputeChannel<Server>;

static RUNTIME: ComputeRuntime<CudaDevice, Server, Channel> = ComputeRuntime::new();

const MEMORY_OFFSET_ALIGNMENT: usize = 32;

/// Initialize a client on the given device with the given options. This function is useful to configure the runtime options
/// or to pick a different graphics API. On wasm, it is necessary to use [`init_async`] instead.
pub fn init(device: &CudaDevice, options: RuntimeOptions) {
    let client = create_client(device, options);
    RUNTIME.register(device, client)
}

fn create_client(device: &CudaDevice, options: RuntimeOptions) -> ComputeClient<Server, Channel> {
    // To get the supported WMMA featurs, and memory properties, we have to initialize the server immediatly.
    cudarc::driver::result::init().unwrap();
    let device_ptr = cudarc::driver::result::device::get(device.index as i32).unwrap();
    let arch = unsafe {
        let major = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap();
        let minor = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap();
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
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory / 4,
        alignment: MEMORY_OFFSET_ALIGNMENT,
    };

    let memory_management = DynamicMemoryManagement::from_configuration(
        storage,
        mem_properties.clone(),
        options.memory_config,
    );
    let cuda_ctx = CudaContext::new(memory_management, stream, ctx, arch);
    let mut server = CudaServer::new(cuda_ctx);
    let mut client_props = ClientProperties::new(&[Feature::Subcube], mem_properties);
    register_wmma_features(&mut client_props, server.arch_version());

    ComputeClient::new(MutexComputeChannel::new(server), client_props)
}

impl Runtime for CudaRuntime {
    type Compiler = CudaCompiler;
    type Server = CudaServer<DynamicMemoryManagement<CudaStorage>>;

    type Channel = MutexComputeChannel<CudaServer<DynamicMemoryManagement<CudaStorage>>>;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            create_client(device, RuntimeOptions::default())
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

fn register_wmma_features(client_props: &mut ClientProperties<Feature>, arch: u32) {
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
            client_props.register_feature(Feature::Cmma {
                a,
                b,
                c,
                m: 16,
                k: 16,
                n: 16,
            });
            client_props.register_feature(Feature::Cmma {
                a,
                b,
                c,
                m: 32,
                k: 16,
                n: 8,
            });
            client_props.register_feature(Feature::Cmma {
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

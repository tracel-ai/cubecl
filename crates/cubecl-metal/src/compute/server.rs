use super::storage::MetalStorage;
use crate::compute::stream::MetalStream;
use crate::{METAL_DISPATCH_LIMIT, METAL_MEMORY_ALIGNMENT};
use cubecl_common::{CubeDim, future, profile};
use cubecl_core::compute::{CubeTask, DebugInformation};
use cubecl_core::future::DynFut;
use cubecl_core::server::{Bindings, ComputeServer, Handle, ProfilingToken};
use cubecl_core::{CubeCount, ExecutionMode, Feature, MemoryUsage, server};
use cubecl_cpp::MslCompiler;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryDeviceProperties, offset_handles,
};
use cubecl_runtime::storage::{BindingResource, ComputeStorage};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLLibraryOptimizationLevel,
    MTLMathMode,
};
use profile::ProfileDuration;
use server::ProfileError;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MetalCompiledKernel {
    cube_dim: CubeDim,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[derive(Debug)]
pub struct MetalServer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    stream: MetalStream,
    pipelines: HashMap<KernelId, MetalCompiledKernel>,
    compilation_options: CompilationOptions,
    timestamps: TimestampProfiler,
}

unsafe impl Send for MetalServer {}
unsafe impl Sync for MetalServer {}

impl MetalServer {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        compilation_options: CompilationOptions,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
    ) -> Self {
        let stream = MetalStream::new(device.clone(), memory_properties, memory_config);

        Self {
            device,
            stream,
            pipelines: HashMap::with_capacity(METAL_DISPATCH_LIMIT),
            compilation_options,
            timestamps: TimestampProfiler::default(),
        }
    }

    fn compile_new_kernel(
        &mut self,
        kernel: Box<dyn CubeTask<MslCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
        kernel_id: &KernelId,
    ) -> MetalCompiledKernel {
        let _is_atomic = kernel_id.stable_format().contains("atomic");

        let mut jit_kernel =
            kernel.compile(&mut Default::default(), &self.compilation_options, mode);
        let cube_dim = jit_kernel.cube_dim;

        if logger.compilation_activated() {
            jit_kernel.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&jit_kernel.source) {
                jit_kernel.source = formatted;
            }
        }
        logger.log_compilation(&jit_kernel);

        autoreleasepool(|_| {
            let compile_options = MTLCompileOptions::new();

            unsafe {
                compile_options.setMathMode(MTLMathMode::Fast);
                compile_options.setOptimizationLevel(MTLLibraryOptimizationLevel::Default);
            }
            let library = self
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&jit_kernel.source),
                    Some(&compile_options),
                )
                .expect("Failed to create Metal library");

            let function = library
                .newFunctionWithName(&NSString::from_str(&jit_kernel.entrypoint_name))
                .expect("Failed to find function in library");

            let pipeline_state = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .expect("Failed to create compute pipeline state");

            MetalCompiledKernel {
                cube_dim,
                pipeline: pipeline_state,
            }
        })
    }

    pub(crate) fn pipeline(
        &mut self,
        kernel: Box<dyn CubeTask<MslCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> MetalCompiledKernel {
        let mut kernel_id = kernel.id();

        kernel_id.mode(mode);

        if let Some(cached_kernel) = self.pipelines.get(&kernel_id) {
            return cached_kernel.clone();
        }

        let compiled_kernel = self.compile_new_kernel(kernel, mode, logger, &kernel_id);

        self.pipelines
            .insert(kernel_id.clone(), compiled_kernel.clone());

        compiled_kernel
    }
}

impl ComputeServer for MetalServer {
    type Kernel = Box<dyn CubeTask<MslCompiler>>;
    type Info = ();
    type Storage = MetalStorage;
    type Feature = Feature;

    fn read(&mut self, bindings: Vec<server::Binding>) -> DynFut<Vec<Vec<u8>>> {
        self.stream.read_buffers(bindings)
    }

    fn read_tensor(&mut self, bindings: Vec<server::BindingWithMeta>) -> DynFut<Vec<Vec<u8>>> {
        let (expected_sizes, bindings): (Vec<_>, Vec<_>) = bindings
            .into_iter()
            .map(|it| {
                (
                    it.shape.iter().product::<usize>() * it.elem_size,
                    it.binding,
                )
            })
            .unzip();
        let data = self.read(bindings);

        Box::pin(async move {
            let mut data = data.await;

            for (data, expected_size) in data.iter_mut().zip(expected_sizes) {
                data.truncate(expected_size);
            }

            data
        })
    }

    fn sync(&mut self) -> DynFut<()> {
        self.stream.sync()
    }

    fn get_resource(
        &mut self,
        binding: server::Binding,
    ) -> BindingResource<<Self::Storage as ComputeStorage>::Resource> {
        let resource = self.stream.get_resource(binding.clone());

        BindingResource::new(binding, resource)
    }

    fn create(&mut self, data: &[u8]) -> Handle {
        let handle = self.empty(data.len());

        self.stream.copy_to_handle(handle.clone(), data);

        handle
    }

    fn create_tensors(
        &mut self,
        data: Vec<&[u8]>,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let handles_strides: Vec<(Handle, Vec<usize>)> =
            self.empty_tensors(shapes.clone(), elem_sizes);

        for (data_slice, (handle, _)) in data.iter().zip(handles_strides.iter()) {
            self.stream.copy_to_handle(handle.clone(), data_slice);
        }

        handles_strides
    }

    fn empty(&mut self, size: usize) -> Handle {
        self.stream.empty(size)
    }

    fn empty_tensors(
        &mut self,
        shapes: Vec<&[usize]>,
        elem_sizes: Vec<usize>,
    ) -> Vec<(Handle, Vec<usize>)> {
        let strides = shapes
            .iter()
            .map(|shape| contiguous_strides(shape))
            .collect::<Vec<_>>();
        let sizes = shapes
            .iter()
            .map(|it| it.iter().product::<usize>())
            .zip(elem_sizes)
            .map(|(size, elem_size)| (size * elem_size).next_multiple_of(METAL_MEMORY_ALIGNMENT))
            .collect::<Vec<_>>();

        let total_size = sizes.iter().sum::<usize>();

        let mem_handle = self.empty(total_size);
        let handles = offset_handles(mem_handle, &sizes);

        handles.into_iter().zip(strides).collect()
    }

    unsafe fn execute(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        kind: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) {
        let cached_kernel = self.pipeline(kernel, kind, logger);
        let pipeline = cached_kernel.pipeline.clone();
        let cube_dim = cached_kernel.cube_dim;

        self.stream.register(pipeline, cube_dim, bindings, &count);
    }

    fn flush(&mut self) {
        self.stream.flush()
    }

    fn memory_usage(&self) -> MemoryUsage {
        self.stream.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self) {
        self.stream.memory_management.cleanup(true);
    }

    fn start_profile(&mut self) -> ProfilingToken {
        future::block_on(self.sync());

        self.timestamps.start()
    }

    fn end_profile(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        future::block_on(self.sync());

        self.timestamps.stop(token)
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];

    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

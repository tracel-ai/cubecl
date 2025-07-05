use super::storage::{MetalResource, MetalStorage};
use crate::METAL_DISPATCH_LIMIT;
use block2::RcBlock;
use cubecl_common::CubeDim;
use cubecl_core::future::DynFut;
use cubecl_core::server::Handle;
use cubecl_runtime::memory_management::{
    MemoryConfiguration, MemoryDeviceProperties, MemoryManagement,
};
use cubecl_runtime::server;
use cubecl_runtime::server::{Bindings, CubeCount};
use objc2::__framework_prelude::{ProtocolObject, Retained};
use objc2::rc::autoreleasepool;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLSharedEvent, MTLSharedEventListener, MTLSize,
};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub(crate) struct MetalStream {
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    current_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    current_encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    pending_operations: VecDeque<(u64, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
    pub memory_management: MemoryManagement<MetalStorage>,

    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    event_counter: Arc<AtomicU64>,

    tasks_count: usize,
}

impl MetalStream {
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
    ) -> Self {
        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create Metal command queue");

        let event = device
            .newSharedEvent()
            .expect("Failed to create Metal shared event");

        let storage = MetalStorage::new(device.clone(), memory_properties.clone());
        let memory_management =
            MemoryManagement::from_configuration(storage, &memory_properties, memory_config);

        Self {
            command_queue,
            current_buffer: None,
            current_encoder: None,
            pending_operations: VecDeque::with_capacity(METAL_DISPATCH_LIMIT),
            memory_management,
            event,
            event_counter: Arc::new(AtomicU64::new(0)),
            tasks_count: 0,
        }
    }

    pub(crate) fn get_resource(&mut self, binding: server::Binding) -> MetalResource {
        self.memory_management
            .get_resource(
                binding.memory.clone(),
                binding.offset_start,
                binding.offset_end,
            )
            .expect("Failed to find resource")
    }

    pub(crate) fn copy_to_handle(&mut self, handle: Handle, data: &[u8]) {
        let resource = self.get_resource(handle.binding());

        unsafe {
            let contents = resource.buffer().contents();
            let dst_slice = std::slice::from_raw_parts_mut(
                (contents.as_ptr() as *mut u8).add(resource.offset()),
                data.len(),
            );
            dst_slice.copy_from_slice(data);
        }
    }

    pub fn read_buffers(&mut self, bindings: Vec<server::Binding>) -> DynFut<Vec<Vec<u8>>> {
        if bindings.is_empty() {
            return Box::pin(async move { Vec::new() });
        }

        self.flush();

        autoreleasepool(|_| {
            let command_buffer = self
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer for read");

            let (sender, receiver) = async_channel::bounded(1);

            let resources: Vec<_> = bindings
                .iter()
                .map(|binding| {
                    self.memory_management
                        .get_resource(
                            binding.memory.clone(),
                            binding.offset_start,
                            binding.offset_end,
                        )
                        .expect("Failed to get resource")
                })
                .collect();

            let completion_handler = RcBlock::new({
                move |_buffer| {
                    let results: Vec<Vec<u8>> = resources
                        .iter()
                        .map(|resource| unsafe {
                            let contents = resource.buffer().contents();
                            let data_ptr = (contents.as_ptr() as *const u8).add(resource.offset());
                            let data = std::slice::from_raw_parts(data_ptr, resource.size());
                            data.to_vec()
                        })
                        .collect();

                    let _ = sender.try_send(results);
                }
            });
            let block: *const _ = &*completion_handler;

            unsafe {
                command_buffer.addCompletedHandler(block.cast_mut());
            }
            command_buffer.commit();

            Box::pin(async move { receiver.recv().await.expect("Failed to receive data") })
        })
    }

    pub fn sync(&mut self) -> DynFut<()> {
        self.flush();

        if let Some(&(fence, _)) = self.pending_operations.back() {
            let (sender, receiver) = async_channel::bounded(1);
            let listener = MTLSharedEventListener::new();
            let handler = RcBlock::new(
                move |_evt: NonNull<ProtocolObject<dyn MTLSharedEvent>>, value: u64| {
                    if value >= fence {
                        let _ = sender.try_send(());
                    }
                },
            );
            let block: *const _ = &*handler;

            unsafe {
                self.event
                    .notifyListener_atValue_block(&listener, fence, block.cast_mut());
            }
            self.pending_operations.clear();

            return Box::pin(async move {
                receiver.recv().await.unwrap();
            });
        }

        Box::pin(async move {})
    }

    pub fn empty(&mut self, size: usize) -> Handle {
        let handle = self.memory_management.reserve(size as u64, None);

        Handle::new(handle, None, None, size as u64)
    }

    pub fn register(
        &mut self,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        cube_dim: CubeDim,
        bindings: Bindings,
        dispatch: &CubeCount,
    ) {
        // --- Phase 1: Prepare all resources (single pass optimization) ---
        let has_metadata = !bindings.metadata.data.is_empty();

        // Pre-calculate capacity with a single pass through scalars
        let mut scalar_count = 0;
        let mut non_empty_scalars = Vec::new();
        for (elem, scalar) in &bindings.scalars {
            if !scalar.data().is_empty() {
                scalar_count += 1;
                non_empty_scalars.push((*elem, scalar));
            }
        }

        let total_resources = bindings.buffers.len() + has_metadata as usize + scalar_count;
        let mut all_resources = Vec::with_capacity(total_resources);

        // Add buffer resources
        for binding in bindings.buffers {
            all_resources.push(self.get_resource(binding));
        }

        // Add metadata if needed
        if has_metadata {
            let info_data: &[u8] = bytemuck::cast_slice(&bindings.metadata.data);
            let handle = self.empty(info_data.len());
            self.copy_to_handle(handle.clone(), info_data);
            all_resources.push(self.get_resource(handle.binding()));
        }

        // Add scalar resources (already filtered in single pass above)
        for (_elem, scalar) in non_empty_scalars {
            let handle = self.empty(scalar.data().len());
            self.copy_to_handle(handle.clone(), scalar.data());
            all_resources.push(self.get_resource(handle.binding()));
        }

        let dynamic_resource = if let CubeCount::Dynamic(binding) = dispatch {
            Some(self.get_resource(binding.clone()))
        } else {
            None
        };

        // --- Phase 2: Encode and dispatch ---
        if self.current_encoder.is_none() {
            let buffer = self
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            self.current_encoder = Some(
                buffer
                    .computeCommandEncoder()
                    .expect("Command encoder should exist"),
            );
            self.current_buffer = Some(buffer);
        }

        let compute_encoder = self.current_encoder.as_ref().unwrap();
        compute_encoder.setComputePipelineState(&pipeline);

        let threads_per_threadgroup = MTLSize {
            width: cube_dim.x as usize,
            height: cube_dim.y as usize,
            depth: cube_dim.z as usize,
        };

        // Bind resources efficiently
        for (index, resource) in all_resources.iter().enumerate() {
            unsafe {
                compute_encoder.setBuffer_offset_atIndex(
                    Some(resource.buffer()),
                    resource.offset(),
                    index,
                );
            }
        }

        match dispatch {
            CubeCount::Static(x, y, z) => {
                compute_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: *x as usize,
                        height: *y as usize,
                        depth: *z as usize,
                    },
                    threads_per_threadgroup,
                );
            }
            CubeCount::Dynamic(_) => unsafe {
                let metal_resource = dynamic_resource
                    .as_ref()
                    .expect("Dynamic resource should be allocated");

                compute_encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            metal_resource.buffer(),
                            metal_resource.offset(),
                            threads_per_threadgroup,
                        );
            },
        }
        self.tasks_count += 1;

        if self.tasks_count >= METAL_DISPATCH_LIMIT {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if let (Some(buffer), Some(encoder)) =
            (self.current_buffer.take(), self.current_encoder.take())
        {
            encoder.endEncoding();

            let fence = self.event_counter.fetch_add(1, Ordering::SeqCst) + 1;

            buffer.encodeSignalEvent_value(self.event.as_ref(), fence);
            buffer.commit();

            self.pending_operations.push_back((fence, buffer));
            self.tasks_count = 0;
        }
    }
}

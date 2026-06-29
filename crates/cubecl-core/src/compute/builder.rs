use alloc::vec::Vec;
use core::sync::atomic::{AtomicI8, Ordering};
use derive_more::Deref;

use crate::{
    BufferInfo, KernelExpansion, KernelIntegrator, KernelSettings, ScalarInfo,
    ir::{Id, Type},
    prelude::KernelDefinition,
};
use alloc::collections::BTreeMap;
use cubecl_ir::{
    DeviceProperties, FlopCountProcessor, Scope, StorageType, TargetProperties, Value,
};
use cubecl_runtime::config::{
    CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel,
};

/// Prepare a kernel to create a [`KernelDefinition`].
#[derive(Deref)]
pub struct KernelBuilder {
    /// Cube [scope](Scope).
    #[deref]
    pub scope: Scope,
    buffers: Vec<BufferInfo>,
    scalars: BTreeMap<StorageType, usize>,
    tensor_maps: Vec<BufferInfo>,
}

static DEBUG: AtomicI8 = AtomicI8::new(-1);

impl KernelBuilder {
    /// Register a scalar and return the [element](Value) to be used for kernel expansion.
    pub fn scalar(&mut self, storage: StorageType) -> Id {
        let current_id = self.scalars.entry(storage).or_default();
        let id = *current_id;
        *current_id += 1;
        id as Id
    }

    fn buffer_id(&self) -> Id {
        self.buffers.len() as Id + self.tensor_maps.len() as Id
    }

    /// Register a buffer and return the [element](Value) to be used for kernel expansion.
    pub fn buffer(&mut self, value_ty: Type) -> Value {
        let id = self.buffer_id();
        let value = self.scope.global(id, value_ty);
        self.buffers.push(BufferInfo {
            id,
            value,
            has_extended_meta: false,
        });
        value
    }

    /// Register a tensor and return the [element](Value) to be used for kernel expansion.
    pub fn tensor(&mut self, value_ty: Type) -> Value {
        let id = self.buffer_id();
        let value = self.scope.global(id, value_ty);
        self.buffers.push(BufferInfo {
            id,
            value,
            has_extended_meta: true,
        });
        value
    }

    /// Register a tensor map and return the [element](Value) to be used for kernel expansion.
    pub fn tensor_map(&mut self) -> Value {
        let id = self.buffer_id();
        let value = self.scope.tensor_map(id);
        self.tensor_maps.push(BufferInfo {
            id,
            value,
            has_extended_meta: true,
        });
        value
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace(&mut self, position: Id) -> Value {
        let input = self.buffers.get_mut(position as usize);
        input.expect("Position valid").value
    }

    pub fn runtime_properties(&mut self, properties: TargetProperties) {
        self.scope.state_mut().target_properties = properties;
    }

    pub fn device_properties(&mut self, properties: &DeviceProperties) {
        self.scope.device_properties(properties);
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(mut self, settings: KernelSettings) -> KernelDefinition {
        if self.profile.enabled {
            self.buffer(FlopCountProcessor::flop_counter_type());
        }

        let scalars = self
            .scalars
            .into_iter()
            .map(|(ty, count)| ScalarInfo { ty, count })
            .collect();
        KernelIntegrator::new(KernelExpansion {
            scope: self.scope,
            buffers: self.buffers,
            scalars,
            tensor_maps: self.tensor_maps,
        })
        .integrate(settings)
    }

    pub fn new() -> Self {
        let debug = DEBUG.load(Ordering::Relaxed);
        let debug = if debug == -1 {
            let val = match CubeClRuntimeConfig::get().compilation.logger.level {
                CompilationLogLevel::Full => 1,
                _ => 0,
            };

            DEBUG.store(val, Ordering::Relaxed);
            val == 1
        } else {
            debug == 1
        };

        let profile = CubeClRuntimeConfig::get().profiling.hardware_metrics;

        Self {
            scope: Scope::root(debug, profile),
            buffers: Default::default(),
            scalars: Default::default(),
            tensor_maps: Default::default(),
        }
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

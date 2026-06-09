use alloc::vec::Vec;
use core::sync::atomic::{AtomicI8, Ordering};

use crate::{
    BufferInfo, KernelExpansion, KernelIntegrator, KernelSettings, ScalarInfo,
    ir::{Id, Type},
    prelude::KernelDefinition,
};
use alloc::collections::BTreeMap;
use cubecl_ir::{DeviceProperties, Scope, StorageType, TargetProperties, Variable, VariableKind};
use cubecl_runtime::config::{
    CubeClRuntimeConfig, RuntimeConfig, compilation::CompilationLogLevel,
};

/// Prepare a kernel to create a [`KernelDefinition`].
pub struct KernelBuilder {
    /// Cube [scope](Scope).
    pub scope: Scope,
    buffers: Vec<BufferInfo>,
    scalars: BTreeMap<StorageType, usize>,
    tensor_maps: Vec<BufferInfo>,
}

static DEBUG: AtomicI8 = AtomicI8::new(-1);

impl KernelBuilder {
    /// Register a scalar and return the [element](Variable) to be used for kernel expansion.
    pub fn scalar(&mut self, storage: StorageType) -> Variable {
        let id = self.scalars.entry(storage).or_default();
        let expand = self.scope.scalar(*id as Id, storage);
        *id += 1;
        expand
    }

    fn buffer_id(&self) -> Id {
        self.buffers.len() as Id + self.tensor_maps.len() as Id
    }

    /// Register a buffer and return the [element](Variable) to be used for kernel expansion.
    pub fn buffer(&mut self, item: Type) -> Variable {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            has_extended_meta: false,
        });
        self.scope.global(id, item)
    }

    /// Register a tensor and return the [element](Variable) to be used for kernel expansion.
    pub fn tensor(&mut self, item: Type) -> Variable {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            has_extended_meta: true,
        });
        self.scope.global(id, item)
    }

    /// Register a tensor map and return the [element](Variable) to be used for kernel expansion.
    pub fn tensor_map(&mut self, item: Type) -> Variable {
        let id = self.buffer_id();
        self.tensor_maps.push(BufferInfo {
            id,
            item,
            has_extended_meta: true,
        });
        Variable::new(VariableKind::TensorMap(id), item)
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace(&mut self, position: Id) -> Variable {
        let input = self
            .buffers
            .get_mut(position as usize)
            .expect("Position valid");

        self.scope.global(position, input.item)
    }

    pub fn runtime_properties(&mut self, properties: TargetProperties) {
        self.scope.state_mut().target_properties = properties;
    }

    pub fn device_properties(&mut self, properties: &DeviceProperties) {
        self.scope.device_properties(properties);
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self, settings: KernelSettings) -> KernelDefinition {
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

        Self {
            scope: Scope::root(debug),
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

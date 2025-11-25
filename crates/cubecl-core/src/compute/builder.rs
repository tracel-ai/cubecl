use std::{
    rc::Rc,
    sync::atomic::{AtomicI8, Ordering},
};

use alloc::collections::BTreeMap;

use cubecl_ir::{ExpandElement, Scope, StorageType, TargetProperties, Variable, VariableKind};
use cubecl_runtime::{
    config::{GlobalConfig, compilation::CompilationLogLevel},
    kernel::Visibility,
};

use crate::ir::{Id, Type};
use crate::prelude::KernelDefinition;
use crate::{BufferInfo, KernelSettings, ScalarInfo};
use crate::{KernelExpansion, KernelIntegrator};

/// Prepare a kernel to create a [kernel definition](crate::KernelDefinition).
pub struct KernelBuilder {
    /// Cube [scope](Scope).
    pub scope: Scope,
    buffers: Vec<BufferInfo>,
    scalars: BTreeMap<StorageType, usize>,
    tensor_maps: Vec<BufferInfo>,
}

static DEBUG: AtomicI8 = AtomicI8::new(-1);

impl KernelBuilder {
    /// Register a scalar and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn scalar(&mut self, storage: StorageType) -> ExpandElement {
        let id = self.scalars.entry(storage).or_default();
        let expand = self.scope.scalar(*id as Id, storage);
        *id += 1;
        expand
    }

    fn buffer_id(&self) -> Id {
        self.buffers.len() as Id + self.tensor_maps.len() as Id
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_tensor(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::ReadWrite,
            has_extended_meta: true,
        });
        self.scope.output(id, item)
    }

    /// Register a tensor map and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_tensor_map(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.tensor_maps.push(BufferInfo {
            id,
            item,
            visibility: Visibility::ReadWrite,
            has_extended_meta: true,
        });
        ExpandElement::Plain(Variable::new(VariableKind::TensorMapInput(id), item))
    }

    /// Register a tensor map and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_tensor_map(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.tensor_maps.push(BufferInfo {
            id,
            item,
            visibility: Visibility::Read,
            has_extended_meta: true,
        });
        ExpandElement::Plain(Variable::new(VariableKind::TensorMapOutput(id), item))
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_tensor(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::Read,
            has_extended_meta: true,
        });
        self.scope.input(id, item)
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_array(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::ReadWrite,
            has_extended_meta: false,
        });
        self.scope.output(id, item)
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace_output(&mut self, position: Id) -> ExpandElement {
        let input = self
            .buffers
            .get_mut(position as usize)
            .expect("Position valid");

        input.visibility = Visibility::ReadWrite;
        self.scope.input(position, input.item)
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_array(&mut self, item: Type) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::Read,
            has_extended_meta: false,
        });
        self.scope.input(id, item)
    }

    pub fn runtime_properties(&mut self, properties: TargetProperties) {
        self.scope.runtime_properties = Rc::new(properties);
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
            let val = match GlobalConfig::get().compilation.logger.level {
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

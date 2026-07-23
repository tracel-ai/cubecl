use core::sync::atomic::{AtomicI8, Ordering};
use derive_more::Deref;
use pliron::r#type::TypeHandle;
use std::vec::Vec;

use crate::{KernelExpansion, KernelIntegrator, prelude::KernelDefinition};
use alloc::collections::BTreeMap;
use cubecl_ir::{
    DeviceProperties, ElemType, Scope, TargetProperties,
    metadata::{INFO_ALIGN, Info, Metadata, SizedInfoField},
    pliron::value::Value,
    settings::KernelSettings,
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
    scalars: BTreeMap<ElemType, usize>,
    buffer_idx: usize,
    ext_meta_idx: usize,
    settings: KernelSettings,
}

static DEBUG: AtomicI8 = AtomicI8::new(-1);

impl KernelBuilder {
    /// Register a scalar and return the [element](Value) to be used for kernel expansion.
    pub fn scalar(&mut self, storage: ElemType) -> usize {
        let current_id = self.scalars.entry(storage).or_default();
        let id = *current_id;
        *current_id += 1;
        id
    }

    fn inc_buffer_id(&mut self) -> usize {
        let id = self.buffer_idx;
        self.buffer_idx += 1;
        id
    }

    fn inc_ext_meta_id(&mut self) -> usize {
        let id = self.ext_meta_idx;
        self.ext_meta_idx += 1;
        id
    }

    /// Register a buffer and return the [element](Value) to be used for kernel expansion.
    pub fn buffer(&mut self, value_ty: TypeHandle) -> Value {
        let id = self.inc_buffer_id();
        self.scope.global(id, None, value_ty)
    }

    /// Register a tensor and return the [element](Value) to be used for kernel expansion.
    pub fn tensor(&mut self, value_ty: TypeHandle) -> Value {
        let id = self.inc_buffer_id();
        let ext_id = self.inc_ext_meta_id();
        self.scope.global(id, Some(ext_id), value_ty)
    }

    /// Register a tensor map and return the [element](Value) to be used for kernel expansion.
    pub fn tensor_map(&mut self) -> Value {
        let id = self.inc_buffer_id();
        let ext_id = self.inc_ext_meta_id();
        self.scope.tensor_map(id, ext_id)
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace(&mut self, position: usize) -> Value {
        self.scope.kernel_arg(position)
    }

    pub fn runtime_properties(&mut self, properties: TargetProperties) {
        self.scope.state_mut().target_properties = properties;
    }

    pub fn device_properties(&mut self, properties: &DeviceProperties) {
        self.scope.device_properties(properties);
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self) -> KernelDefinition {
        let info = self.create_info();
        KernelIntegrator::new(KernelExpansion {
            scope: self.scope,
            info,
        })
        .integrate(self.settings)
    }

    fn create_info(&self) -> Info {
        let address_type = self.settings.address_type;
        let metadata = Metadata::new(self.buffer_idx, self.ext_meta_idx);
        let mut scalar_fields = Vec::with_capacity(self.scalars.len());
        let mut sized_meta = None;

        let mut offset = 0;

        for (&ty, &count) in self.scalars.iter() {
            scalar_fields.push(SizedInfoField { ty, count, offset });
            offset += (ty.expand_size(address_type) * count).next_multiple_of(INFO_ALIGN);
        }

        if metadata.static_len() > 0 {
            let size = metadata.static_len();
            sized_meta = Some(SizedInfoField {
                ty: address_type.unsigned_type(),
                count: size,
                offset,
            });
            offset += (address_type.size() * size).next_multiple_of(INFO_ALIGN);
        }

        Info {
            scalars: scalar_fields,
            sized_meta,
            has_dynamic_meta: metadata.num_extended_meta() > 0,
            dynamic_meta_offset: offset,
            metadata,
        }
    }

    pub fn new(mut settings: KernelSettings) -> Self {
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
        settings.debug_symbols |= debug;

        Self {
            scope: Scope::root(settings.clone()),
            scalars: Default::default(),
            settings,
            buffer_idx: 0,
            ext_meta_idx: 0,
        }
    }
}

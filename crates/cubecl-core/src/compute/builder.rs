use alloc::vec::Vec;
use core::sync::atomic::{AtomicI8, Ordering};
use derive_more::Deref;
use pliron::r#type::{TypeHandle, type_cast};

use crate::{KernelExpansion, KernelIntegrator, prelude::KernelDefinition};
use alloc::collections::BTreeMap;
use cubecl_ir::{
    DeviceProperties, ElemType, Scope, TargetProperties,
    attributes::{ATTR_BUFFER_BINDING, BufferBindingAttr, FuncInterface},
    features::Features,
    interfaces::{ScalarType, TypedExt as _},
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
        self.validate_buffer_types();
        for warning in self.scope.pop_warnings() {
            log::warn!("[{}] {warning}", self.settings.kernel_name);
        }
        let info = self.create_info();
        KernelIntegrator::new(KernelExpansion {
            scope: self.scope,
            info,
        })
        .integrate(self.settings)
    }

    /// Refuse a kernel that declares a buffer whose element type this device cannot store.
    ///
    /// A native size is not a storage representation: `bool` is one byte, yet WGSL
    /// has no host-shareable `bool` buffer and SPIR-V has no physical bool buffer.
    /// Kernels built without device properties (hand-written definitions) are left
    /// alone; everything else is checked against the table the device declared.
    fn validate_buffer_types(&self) {
        let Some(props) = self.scope.state().device_properties.clone() else {
            return;
        };
        refuse_unstorable_buffers(&self.scope, &props.features);
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

/// Refuse a kernel that declares a buffer whose element type `features` cannot
/// store, naming the type and what to do instead.
///
/// Split out from [`KernelBuilder::validate_buffer_types`] so it can be exercised
/// without a device: it needs only a scope and a feature table.
/// Record a validation error for every buffer whose element type `features`
/// cannot store.
///
/// The error goes through [`Scope::push_error`], which every backend already
/// turns into a compilation failure (WGSL, SPIR-V, C++ and LLVM all read the
/// scope's errors), so a refused kernel is reported as a validation failure
/// rather than an expansion panic. Split out from
/// [`KernelBuilder::validate_buffer_types`] so it can be exercised without a
/// device: it needs only a scope and a feature table.
fn refuse_unstorable_buffers(scope: &Scope, features: &Features) {
    let ctx = scope.ctx();
    let entry = scope.state().entry_func;
    let num_args = entry.get_entry_block(ctx).deref(ctx).get_num_arguments();

    for pos in 0..num_args {
        if entry
            .get_arg_attr::<BufferBindingAttr>(ctx, pos, &ATTR_BUFFER_BINDING)
            .is_none()
        {
            continue;
        }
        let arg = entry.get_entry_block(ctx).deref(ctx).get_argument(pos);
        // A bool vector's element type is a vector, so take the *scalar* element
        // through the type interfaces instead of assuming a scalar already.
        let Some(element) = arg.try_get_scalar_elem_ty(ctx) else {
            continue;
        };
        let element = element.deref(ctx);
        let Some(scalar) = type_cast::<dyn ScalarType>(&*element) else {
            continue;
        };
        let elem = scalar.elem_type(ctx);
        if features.supports_buffer(elem) {
            continue;
        }

        scope.push_error(alloc::format!(
            "this device cannot store a buffer of {elem:?}: its feature table does not \
             register that element type for buffer usage. Bind the data as a storable \
             element type instead (for `bool`, a byte or word per element)."
        ));
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_ir::{
        AddressType,
        features::TypeUsage,
        settings::{Dim3, ExecutionMode},
        types::{VectorType, scalar::BoolType},
    };

    fn settings() -> KernelSettings {
        KernelSettings::new(
            Dim3 { x: 1, y: 1, z: 1 },
            ExecutionMode::Checked,
            AddressType::U32,
        )
    }

    fn scope_with(kernel: impl FnOnce(&mut KernelBuilder)) -> Scope {
        let mut builder = KernelBuilder::new(settings());
        kernel(&mut builder);
        builder.scope
    }

    fn bool_buffer(builder: &mut KernelBuilder) {
        let ty = BoolType::get(builder.scope.ctx_mut()).to_handle();
        let _ = builder.buffer(ty);
    }

    fn features_with(bool_storable: bool) -> Features {
        let mut features = Features::default();
        let uses = if bool_storable {
            TypeUsage::all()
        } else {
            TypeUsage::all() - TypeUsage::Buffer
        };
        features.types.elem.insert(ElemType::Bool, uses);
        features
    }

    #[test]
    fn a_bool_buffer_is_refused_where_the_device_cannot_store_it() {
        let scope = scope_with(bool_buffer);
        refuse_unstorable_buffers(&scope, &features_with(false));
        let errors = scope.pop_errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].contains("cannot store a buffer of Bool"),
            "{errors:?}"
        );
    }

    #[test]
    fn a_bool_buffer_is_allowed_where_the_device_can_store_it() {
        let scope = scope_with(bool_buffer);
        refuse_unstorable_buffers(&scope, &features_with(true));
        assert!(scope.pop_errors().is_empty());
    }

    #[test]
    fn a_vector_of_bool_is_checked_too() {
        let scope = scope_with(|builder| {
            let ty = BoolType::get(builder.scope.ctx_mut()).to_handle();
            let vector = VectorType::get(builder.scope.ctx_mut(), ty, 4).to_handle();
            let _ = builder.buffer(vector);
        });
        refuse_unstorable_buffers(&scope, &features_with(false));
        assert_eq!(scope.pop_errors().len(), 1);
    }

    #[test]
    fn a_kernel_without_device_properties_is_left_alone() {
        let scope = scope_with(bool_buffer);
        assert!(scope.pop_errors().is_empty());
    }
}

use cubecl_common::CubeDim;
use cubecl_ir::{Elem, Id, Item, Scope};

use crate::{
    compute::{Binding, KernelDefinition, Location, ScalarBinding, Visibility},
    prelude::FastMath,
};

/// The kernel integrator allows you to create a [kernel definition](KernelDefinition) based on
/// [kernel expansion](KernelExpansion) and [kernel settings](KernelSettings).
#[derive(Clone)]
pub struct KernelIntegrator {
    expansion: KernelExpansion,
    buffer_bindings: Vec<Binding>,
    scalar_bindings: Vec<ScalarBinding>,
    tensor_maps: Vec<Id>,
}

/// The information necessary to compile a [kernel definition](KernelDefinition).
#[derive(Clone)]
pub struct KernelExpansion {
    pub buffers: Vec<BufferInfo>,
    pub scalars: Vec<ScalarInfo>,
    pub tensor_maps: Vec<Id>,
    pub scope: Scope,
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
pub struct KernelSettings {
    pub cube_dim: CubeDim,
    pub options: KernelOptions,
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
pub struct KernelOptions {
    pub kernel_name: String,
    pub debug_symbols: bool,
    pub fp_math_mode: FastMath,
    pub cluster_dim: Option<CubeDim>,
}

impl KernelSettings {
    /// Set cube dimension.
    #[allow(dead_code)]
    pub fn cube_dim(mut self, cube_dim: CubeDim) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// Set kernel name.
    #[allow(dead_code)]
    pub fn kernel_name<S: AsRef<str>>(mut self, name: S) -> Self {
        self.options.kernel_name = name.as_ref().to_string();
        self
    }

    /// Activate debug symbols
    pub fn debug_symbols(mut self) -> Self {
        self.options.debug_symbols = true;
        self
    }

    /// Set FP math mode
    pub fn fp_math_mode(mut self, mode: FastMath) -> Self {
        self.options.fp_math_mode = mode;
        self
    }

    /// Set cluster dim
    pub fn cluster_dim(mut self, cluster_dim: CubeDim) -> Self {
        self.options.cluster_dim = Some(cluster_dim);
        self
    }
}

/// Information related to a buffer binding.
#[derive(Clone, Debug)]
pub struct BufferInfo {
    pub id: Id,
    pub item: Item,
    pub visibility: Visibility,
    /// Whether this input has extended metadata (rank, shape, strides)
    pub has_extended_meta: bool,
}

/// Information related to a scalar input.
#[derive(Clone, Debug)]
pub struct ScalarInfo {
    pub elem: Elem,
    pub count: usize,
}

impl KernelIntegrator {
    /// Starts a new compilation.
    pub fn new(info: KernelExpansion) -> Self {
        Self {
            expansion: info,
            buffer_bindings: Default::default(),
            scalar_bindings: Default::default(),
            tensor_maps: Default::default(),
        }
    }

    /// Performs the compilation with the provided [settings](KernelSettings).
    pub fn integrate(mut self, settings: KernelSettings) -> KernelDefinition {
        self.register_buffers();
        self.register_scalars();
        self.register_tensor_maps();

        self.scalar_bindings.sort_by_key(|binding| binding.elem);

        KernelDefinition {
            buffers: self.buffer_bindings,
            tensor_maps: self.tensor_maps,
            scalars: self.scalar_bindings,
            cube_dim: settings.cube_dim,
            body: self.expansion.scope,
            options: settings.options,
        }
    }

    fn register_buffers(&mut self) {
        for buffer in self.expansion.buffers.drain(..) {
            self.buffer_bindings.push(Binding {
                id: buffer.id,
                item: buffer.item,
                visibility: buffer.visibility,
                location: Location::Storage,
                has_extended_meta: buffer.has_extended_meta,
                size: None,
            });
        }
    }

    fn register_scalars(&mut self) {
        for scalar in self.expansion.scalars.drain(..) {
            self.scalar_bindings.push(ScalarBinding {
                elem: scalar.elem,
                count: scalar.count,
            });
        }
    }

    fn register_tensor_maps(&mut self) {
        for id in self.expansion.tensor_maps.drain(..) {
            self.tensor_maps.push(id);
        }
    }
}

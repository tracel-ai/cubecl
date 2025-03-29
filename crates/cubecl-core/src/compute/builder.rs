use cubecl_ir::{ExpandElement, Scope, Variable, VariableKind};
use cubecl_runtime::debug::DebugLogger;

use crate::ir::{Elem, Id, Item};
use crate::prelude::KernelDefinition;
use crate::{BufferInfo, KernelSettings, ScalarInfo};
use crate::{KernelExpansion, KernelIntegrator};
use hashbrown::HashMap;

use super::Visibility;

/// Prepare a kernel to create a [kernel definition](crate::KernelDefinition).
pub struct KernelBuilder {
    /// Cube [scope](Scope).
    pub context: Scope,
    buffers: Vec<BufferInfo>,
    scalars: HashMap<Elem, usize>,
    tensor_maps: Vec<Id>,
}

impl KernelBuilder {
    /// Register a scalar and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn scalar(&mut self, elem: Elem) -> ExpandElement {
        let id = self.scalars.entry(elem).or_default();
        let expand = self.context.scalar(*id as Id, elem);
        *id += 1;
        expand
    }

    fn buffer_id(&self) -> Id {
        self.buffers.len() as Id + self.tensor_maps.len() as Id
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_tensor(&mut self, item: Item) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::ReadWrite,
            has_extended_meta: true,
        });
        self.context.output(id, item)
    }

    /// Register a tensor map and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn tensor_map(&mut self) -> ExpandElement {
        let id = self.buffer_id();
        self.tensor_maps.push(id);
        ExpandElement::Plain(Variable::new(
            VariableKind::TensorMap(id),
            Item::new(Elem::Bool),
        ))
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_tensor(&mut self, item: Item) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::Read,
            has_extended_meta: true,
        });
        self.context.input(id, item)
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_array(&mut self, item: Item) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::ReadWrite,
            has_extended_meta: false,
        });
        self.context.output(id, item)
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace_output(&mut self, position: Id) -> ExpandElement {
        let input = self
            .buffers
            .get_mut(position as usize)
            .expect("Position valid");

        input.visibility = Visibility::ReadWrite;
        self.context.input(position, input.item)
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_array(&mut self, item: Item) -> ExpandElement {
        let id = self.buffer_id();
        self.buffers.push(BufferInfo {
            id,
            item,
            visibility: Visibility::Read,
            has_extended_meta: false,
        });
        self.context.input(id, item)
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self, settings: KernelSettings) -> KernelDefinition {
        let scalars = self
            .scalars
            .into_iter()
            .map(|(elem, count)| ScalarInfo { elem, count })
            .collect();
        KernelIntegrator::new(KernelExpansion {
            scope: self.context,
            buffers: self.buffers,
            scalars,
            tensor_maps: self.tensor_maps,
        })
        .integrate(settings)
    }

    pub fn new() -> Self {
        Self {
            context: Scope::root(DebugLogger::default().is_activated()),
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

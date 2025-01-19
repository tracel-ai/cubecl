use cubecl_ir::ExpandElement;

use crate::ir::{Elem, Id, Item};
use crate::prelude::KernelDefinition;
use crate::KernelSettings;
use crate::{frontend::CubeContext, InputInfo, KernelExpansion, KernelIntegrator, OutputInfo};
use std::collections::HashMap;

use super::Visibility;

/// Prepare a kernel to create a [kernel definition](crate::KernelDefinition).
pub struct KernelBuilder {
    /// Cube [context](CubeContext).
    pub context: CubeContext,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    indices: HashMap<Elem, usize>,
    num_input: Id,
    num_output: Id,
}

impl KernelBuilder {
    /// Register a scalar and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn scalar(&mut self, elem: Elem) -> ExpandElement {
        let index = match self.indices.get_mut(&elem) {
            Some(index) => match self.inputs.get_mut(*index).unwrap() {
                InputInfo::Scalar { elem: _, size } => {
                    *size += 1;
                    *size as Id - 1
                }
                _ => panic!("Should be a scalar."),
            },
            None => {
                self.indices.insert(elem, self.inputs.len());
                self.inputs.push(InputInfo::Scalar { size: 1, elem });
                0
            }
        };

        self.context.scalar(index, elem)
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_tensor(&mut self, item: Item) -> ExpandElement {
        self.outputs.push(OutputInfo::Array {
            item,
            has_extended_meta: true,
        });
        let variable = self.context.output(self.num_output, item);
        self.num_output += 1;

        variable
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_tensor(&mut self, item: Item) -> ExpandElement {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: Visibility::Read,
            has_extended_meta: true,
        });
        let variable = self.context.input(self.num_input, item);
        self.num_input += 1;
        variable
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_array(&mut self, item: Item) -> ExpandElement {
        self.outputs.push(OutputInfo::Array {
            item,
            has_extended_meta: false,
        });
        let variable = self.context.output(self.num_output, item);
        self.num_output += 1;

        variable
    }

    /// Register an output that uses the same resource as the input as the given position.
    pub fn inplace_output(&mut self, position: Id) -> ExpandElement {
        let input = self
            .inputs
            .get_mut(position as usize)
            .expect("Position valid");

        if let InputInfo::Array {
            visibility, item, ..
        } = input
        {
            *visibility = Visibility::ReadWrite;
            let variable = self.context.input(position, *item);
            return variable;
        }

        panic!("No input found at position {position}");
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_array(&mut self, item: Item) -> ExpandElement {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: Visibility::Read,
            has_extended_meta: false,
        });
        let variable = self.context.input(self.num_input, item);
        self.num_input += 1;
        variable
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self, settings: KernelSettings) -> KernelDefinition {
        KernelIntegrator::new(KernelExpansion {
            scope: self.context.into_scope(),
            inputs: self.inputs,
            outputs: self.outputs,
            kernel_name: settings.kernel_name.clone(),
        })
        .integrate(settings)
    }

    pub fn new() -> Self {
        Self {
            context: CubeContext::root(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            indices: HashMap::new(),
            num_input: 0,
            num_output: 0,
        }
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

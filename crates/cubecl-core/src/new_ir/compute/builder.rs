use crate::{
    frontend::CubeContext, new_ir::Expression, InputInfo, KernelExpansion, KernelIntegrator,
    OutputInfo,
};
use crate::{
    ir::{Elem, Item, Visibility},
    new_ir::Primitive,
};
use crate::{new_ir::GlobalVariable, prelude::KernelDefinition};
use crate::{new_ir::SquareType, KernelSettings};
use std::{collections::HashMap, num::NonZero};

use super::flatten::flatten_expr;

/// Prepare a kernel to create a [kernel definition](crate::KernelDefinition).
pub struct KernelBuilder {
    /// Cube [context](CubeContext).
    pub context: CubeContext,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    indices: HashMap<Elem, usize>,
    num_input: u16,
    num_output: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GlobalType {
    Scalar,
    InputArray,
    OutputArray,
}

impl KernelBuilder {
    /// Register a scalar and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn scalar<T: Primitive>(&mut self, elem: Elem) -> GlobalVariable<T> {
        let index = match self.indices.get_mut(&elem) {
            Some(index) => match self.inputs.get_mut(*index).unwrap() {
                InputInfo::Scalar { elem: _, size } => {
                    *size += 1;
                    *size as u16 - 1
                }
                _ => panic!("Should be a scalar."),
            },
            None => {
                self.indices.insert(elem, self.inputs.len());
                self.inputs.push(InputInfo::Scalar { size: 1, elem });
                0
            }
        };

        GlobalVariable::new(index, GlobalType::Scalar, None)
    }

    /// Register an output array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn output_array<T: SquareType>(&mut self, item: Item) -> GlobalVariable<T> {
        self.outputs.push(OutputInfo::Array { item });
        let variable = GlobalVariable::new(
            self.num_output,
            GlobalType::OutputArray,
            NonZero::new(item.vectorization),
        );
        self.num_output += 1;

        variable
    }

    /// Register an input array and return the [element](ExpandElement) to be used for kernel expansion.
    pub fn input_array<T: SquareType>(&mut self, item: Item) -> GlobalVariable<T> {
        self.inputs.push(InputInfo::Array {
            item,
            visibility: Visibility::Read,
        });
        let variable = GlobalVariable::new(
            self.num_input,
            GlobalType::InputArray,
            NonZero::new(item.vectorization),
        );
        self.num_input += 1;
        variable
    }

    pub fn apply_expansion(&mut self, expr: Expression) {
        flatten_expr(expr, &mut self.context);
    }

    /// Build the [kernel definition](KernelDefinition).
    pub fn build(self, settings: KernelSettings) -> KernelDefinition {
        KernelIntegrator::new(KernelExpansion {
            scope: self.context.into_scope(),
            inputs: self.inputs,
            outputs: self.outputs,
        })
        .integrate(settings)
    }
}

impl Default for KernelBuilder {
    fn default() -> Self {
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

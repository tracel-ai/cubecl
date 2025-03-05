/*
Corresponds to wgpu/compiler/shader.rs. The compiled executable kernel functions, stored
as dynamically linked libraries.
*/

use alloc::fmt::Display;

use cranelift::prelude::{types, EntityRef, FunctionBuilderContext, Signature};
use cranelift::prelude::{FunctionBuilder, Variable as CraneliftVariable};
use cranelift_codegen::ir::{types::Type, AbiParam, Function, UserFuncName};
use cubecl_core::ir::Item;
use cubecl_core::{
    compute::Binding,
    ir::{Elem, FloatKind, IntKind, UIntKind},
    prelude::KernelDefinition,
};
use hashbrown::HashMap;

use super::{to_ssa_type, CompilerState, LookupTables};

pub fn compile_binding(binding: &Binding) -> AbiParam {
    let mut vtype = to_ssa_type(&binding.item);

    AbiParam {
        value_type: vtype,
        purpose: cranelift_codegen::ir::ArgumentPurpose::Normal,
        extension: cranelift_codegen::ir::ArgumentExtension::None,
    }
}

impl<'a> CompilerState<'a> {
    pub(crate) fn create_function(
        lookup: &mut LookupTables,
        kernel: &KernelDefinition,
    ) -> Function {
        //TODO: calling convention needs to be changable, likely at compile time
        let mut signature = Signature::new(cranelift_codegen::isa::CallConv::Fast);
        signature.params = kernel.inputs.iter().map(compile_binding).collect();

        signature.returns = kernel.outputs.iter().map(compile_binding).collect();
        //TODO: need to add a symbol table for mapping names
        let sym = lookup.insert_func(kernel.options.kernel_name.clone());
        let res = Function::with_name_signature(UserFuncName::user(sym.0, sym.1), signature);
        res
    }
    pub(crate) fn declare_variables(&mut self, kernel: &KernelDefinition) {}
    //TODO: apparently named is not what I need to use here
    // pub(crate) fn init_variables(
    //     &mut self,
    //     builder: &mut FunctionBuilder,
    //     kernel: &KernelDefinition,
    // ) -> HashMap<String, CraneliftVariable> {

    //     kernel.inputs.iter().for_each(|binding|{
    //         let var = builder.getsert()
    //     });
    // }
    pub(crate) fn translate_scope(&mut self, kernel: &KernelDefinition) {
        todo!()
    }
}

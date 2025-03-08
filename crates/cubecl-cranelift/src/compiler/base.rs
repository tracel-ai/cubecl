use alloc::{fmt::Debug, sync::Arc};
use cranelift::prelude::{
    Block, Configurable, EntityRef, FunctionBuilder, FunctionBuilderContext, InstBuilder, Value,
    Variable as CraneliftVariable,
};
use cranelift_codegen::{
    entity::SecondaryMap,
    ir::{BlockData, Function},
    isa::TargetIsa,
    settings, Context,
};

use cubecl_core::{
    ir::{Item, VariableKind},
    prelude::KernelDefinition,
    Compiler, ExecutionMode,
};

use hashbrown::HashMap;

use super::{compile_binding, LookupTables};

pub struct FunctionCompiler {
    //ctx: CompilerState,
    isa: Arc<dyn TargetIsa>,
    exec_mode: ExecutionMode,
    pub(crate) codegen_ctx: Context,
    pub(crate) builder_ctx: FunctionBuilderContext,
}
// need to figure out a better name
pub enum CompilerOutput {
    ///Output is interpreted (Wasm)
    JIT,
    ///Output is a shared library
    LIB,
}

///State for the compiler. This struct is instantiated at the start of compilation of a kernel,
/// and borrows context structs from the compiler (hence the lifetime).
pub struct CompilerState<'a> {
    pub(crate) lookup: LookupTables,
    pub(crate) pointer_type: cranelift_codegen::ir::types::Type,

    pub(crate) func_builder: FunctionBuilder<'a>,
    pub(crate) entry_block: Block,
}

impl Default for FunctionCompiler {
    fn default() -> Self {
        let mut flag_builder = settings::builder();
        //whether to assume libcalls are available in the same module
        //generally seems to be recommended to set to true for JIT and false
        //for AOT/shared libraries.
        flag_builder.set("use_colocated_libcalls", "true").unwrap();
        //not sure if default should be speed or speed and size
        flag_builder.set("opt_level", "speed").unwrap();
        //position-independent code is code not tied to a specific address
        //and is recommended for shared libraries.
        flag_builder.set("is_pic", "true").unwrap();
        let mut isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });

        //currently can't find any documentation on configurable isa flags, will come back to this
        //isa_builder.set()
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let mut codegen_ctx = new_context(&isa);

        Self {
            isa,
            exec_mode: ExecutionMode::default(),
            codegen_ctx,
            builder_ctx: FunctionBuilderContext::new(),
        }
    }
}

fn new_context(isa: &Arc<dyn TargetIsa>) -> Context {
    let mut codegen_ctx = Context::new();
    codegen_ctx.func.signature.call_conv = isa.default_call_conv();
    codegen_ctx
}

impl Compiler for FunctionCompiler {
    type Representation = Function;

    type CompilationOptions = ();

    fn compile(
        &mut self,
        kernel: cubecl_core::prelude::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation {
        let mut state = CompilerState::new(self, &kernel);
        state.process_scope(&kernel);
        todo!()
    }

    fn elem_size(&self, elem: cubecl_core::ir::Elem) -> usize {
        todo!()
    }
}

impl Debug for FunctionCompiler {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionCompiler")
            .field("exec_mode", &self.exec_mode)
            //.field("builder_func", &self.builder.func)
            .finish()
    }
}
impl<'a> CompilerState<'a> {
    pub fn new(compiler: &'a mut FunctionCompiler, kernel: &KernelDefinition) -> Self {
        let mut lookup = LookupTables::default();

        compiler.codegen_ctx.func.signature.params =
            kernel.inputs.iter().map(compile_binding).collect();
        compiler.codegen_ctx.func.signature.returns =
            kernel.outputs.iter().map(compile_binding).collect();

        let mut func_builder =
            FunctionBuilder::new(&mut compiler.codegen_ctx.func, &mut compiler.builder_ctx);
        let entry_block = func_builder.create_block();
        func_builder.append_block_params_for_function_params(entry_block);
        func_builder.switch_to_block(entry_block);
        func_builder.seal_block(entry_block);

        // for those of you who, like the author, are wondering why we are iterating through
        // the values of the inputs to define the function inputs twice, apparently the inputs to a function
        // apparently the earlier calls don't map the inputs to variables within the block
        for i in 0..kernel.inputs.len() {
            let var = lookup.getsert_var(VariableKind::GlobalInputArray(i as u32));
            let val = func_builder.block_params(entry_block)[i];
            func_builder.def_var(var, val)
        }

        Self {
            pointer_type: compiler.isa.pointer_type(),
            lookup,
            func_builder,
            entry_block,
        }
    }
}

impl Clone for FunctionCompiler {
    fn clone(&self) -> Self {
        Self {
            isa: self.isa.clone(),
            exec_mode: self.exec_mode,
            codegen_ctx: new_context(&self.isa),
            builder_ctx: FunctionBuilderContext::new(),
        }
    }
}

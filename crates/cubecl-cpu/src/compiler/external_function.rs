use tracel_llvm::melior::{
    Context, ExecutionEngine,
    dialect::llvm,
    ir::{
        BlockLike, Identifier, Location, Region,
        attribute::{StringAttribute, TypeAttribute},
        r#type::IntegerType,
    },
};

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("printf", libc::printf as *mut ());
        execution_engine.register_symbol("_mlir_printf", libc::printf as *mut ()); // This is only there to fool the execution engine to generate .so for inspection even if symbol resolution will probably not work.
    }
}

pub fn add_external_function_to_module<'a>(
    context: &'a Context,
    module: &tracel_llvm::melior::ir::Module<'a>,
) {
    let integer_type = IntegerType::new(context, 32).into();
    let func_type = TypeAttribute::new(llvm::r#type::function(
        integer_type,
        &[llvm::r#type::pointer(context, 0)],
        true,
    ));
    module.body().append_operation(llvm::func(
        context,
        StringAttribute::new(context, "printf"),
        func_type,
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        Location::unknown(context),
    ));
}

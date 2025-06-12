use tracel_llvm::melior::{
    Context, ExecutionEngine,
    dialect::func,
    ir::{
        BlockLike, Identifier, Location, Region, Type,
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
    },
};

extern "C" fn print_i(index: isize) {
    println!("{index}");
}

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("print_i", print_i as *mut fn(isize) as *mut ());
        execution_engine.register_symbol("_mlir_print_i", print_i as *mut fn(isize) as *mut ()); // This is only there to fool the execution engine to generate .so for inspection even if symbol resolution will probably not work.
    }
}

pub fn add_external_function_to_module<'a>(
    context: &'a Context,
    module: &tracel_llvm::melior::ir::Module<'a>,
) {
    let func_type =
        TypeAttribute::new(FunctionType::new(context, &[Type::index(context)], &[]).into());
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "print_i"),
        func_type,
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        Location::unknown(context),
    ));
}

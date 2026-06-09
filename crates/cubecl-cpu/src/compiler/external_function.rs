use tracel_llvm::mlir_rs::{
    Context, ExecutionEngine,
    dialect::llvm::{
        self,
        attributes::{Linkage, linkage},
    },
    ir::{
        BlockLike, Identifier, Location, Region,
        attribute::{StringAttribute, TypeAttribute},
        r#type::IntegerType,
    },
};

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("rsqrtf", rsqrtf as *mut ());
        execution_engine.register_symbol("rsqrt", rsqrt as *mut ());
    }
}

extern "C" fn rsqrtf(input: f32) -> f32 {
    1.0f32 / input.sqrt()
}

extern "C" fn rsqrt(input: f64) -> f64 {
    1.0f64 / input.sqrt()
}

pub fn add_external_function_to_module<'a>(
    context: &'a Context,
    module: &tracel_llvm::mlir_rs::ir::Module<'a>,
) {
    let location = Location::unknown(context);
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
            Identifier::new(context, "linkage"),
            linkage(context, Linkage::External),
        )],
        location,
    ));
}

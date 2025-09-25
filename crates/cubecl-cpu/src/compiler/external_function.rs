use tracel_llvm::mlir_rs::{
    Context, ExecutionEngine,
    dialect::{
        func,
        llvm::{
            self,
            attributes::{Linkage, linkage},
        },
    },
    ir::{
        BlockLike, Identifier, Location, Region,
        attribute::{StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
    },
};

use crate::compute::compute_task::sync_cube;

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("sync_cube", sync_cube as *mut ());
        // This is only there to fool the execution engine to generate .so for inspection even if symbol resolution will probably not work.
        execution_engine.register_symbol("_mlir_sync_cube", sync_cube as *mut ());
    }
}

pub fn add_external_function_to_module<'a>(
    context: &'a Context,
    module: &tracel_llvm::mlir_rs::ir::Module<'a>,
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
            Identifier::new(context, "linkage"),
            linkage(context, Linkage::External),
        )],
        Location::unknown(context),
    ));
    let func_name = StringAttribute::new(context, "sync_cube");
    let func_type = TypeAttribute::new(FunctionType::new(context, &[], &[]).into());
    module.body().append_operation(func::func(
        context,
        func_name,
        func_type,
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        Location::unknown(context),
    ));
}

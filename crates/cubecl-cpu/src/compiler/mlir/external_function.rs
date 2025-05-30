use melior::{
    ExecutionEngine,
    dialect::func,
    ir::{
        BlockLike, Identifier, Location, Region, Type,
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
    },
};

use super::visitor::Visitor;

extern "C" fn print_i(index: isize) {
    println!("{index}");
}

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("print_i", print_i as *mut fn(isize) as *mut ());
        execution_engine.register_symbol("_mlir_print_i", print_i as *mut fn(isize) as *mut ()); // This is only there to fool the execution engine to generate .so for inspection even if symbol resolution will probably not work.
    }
}

impl<'a> Visitor<'a> {
    pub fn add_external_function_to_module(&self, module: &melior::ir::Module<'a>) {
        let func_type = TypeAttribute::new(
            FunctionType::new(self.context, &[Type::index(self.context)], &[]).into(),
        );
        module.body().append_operation(func::func(
            self.context,
            StringAttribute::new(self.context, "print_i"),
            func_type,
            Region::new(),
            &[(
                Identifier::new(self.context, "sym_visibility"),
                StringAttribute::new(self.context, "private").into(),
            )],
            Location::unknown(self.context),
        ));
    }
}

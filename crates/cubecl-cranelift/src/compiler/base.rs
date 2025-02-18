use cranelift::prelude::FunctionBuilder;
use cranelift_codegen::ir::Function;
use cubecl_core::ExecutionMode;

pub struct FunctionCompiler<'a> {
    builder: FunctionBuilder<'a>,
    exec_mode: ExecutionMode,
}

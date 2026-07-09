pub mod dialect;
pub mod jit;

use cubecl_common::backtrace::BackTrace;
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{Compiler, prelude::*};
#[cfg(feature = "pliron-dump")]
use pliron::{builtin::ops::ModuleOp, context::Context, printable::Printable};

use crate::compiler::jit::engine::PlironEngine;

#[derive(Clone, Debug, Default)]
pub struct PlironCompiler {}

#[derive(Clone, Debug, Default)]
pub struct PlironOptions;

impl Compiler for PlironCompiler {
    type Representation = PlironEngine;

    type CompilationOptions = PlironOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions, // TODO pass this through the visitor, though it doesn't need anything for the moment
    ) -> Result<Self::Representation, CompilationError> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile pliron kernel\n Caused by:\n  ".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        Ok(self.clone().compile_ir(kernel))
    }

    fn extension(&self) -> &'static str {
        "plir"
    }
}

impl PlironCompiler {
    fn compile_ir(self, kernel: KernelDefinition) -> PlironEngine {
        let _module = kernel.body.state().module;
        let mut _ctx = kernel.body.into_context().expect("Should be owned scope");

        #[cfg(feature = "pliron-dump")]
        dump_pliron(&_module, &_ctx, &kernel.settings.kernel_name);

        PlironEngine::default()
    }
}

#[cfg(feature = "pliron-dump")]
fn dump_pliron(module: &ModuleOp, ctx: &Context, name: &str) {
    use std::fs;
    if let Ok(dir) = std::env::var("CUBECL_DEBUG_PLIRON") {
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir(&path);
        std::fs::write(
            format!("{}/initial.plir", path),
            format!("{}", module.disp(&ctx)),
        )
        .unwrap();
    }
}

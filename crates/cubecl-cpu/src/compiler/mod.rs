pub mod jit;

use cubecl_common::backtrace::BackTrace;
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{Compiler, prelude::KernelDefinition};

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
            let mut reason = "Can't compile mlir kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        #[cfg(feature = "mlir-dump")]
        dump_scope(&kernel.body, &kernel.options.kernel_name);

        #[cfg(feature = "mlir-dump")]
        dump_opt(&opt, &kernel.options.kernel_name);
        Ok(PlironEngine::default())
    }

    fn extension(&self) -> &'static str {
        "mlir"
    }
}

#[cfg(feature = "mlir-dump")]
pub fn get_dump_name(name: &str) -> Option<std::path::PathBuf> {
    use std::fs;

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_MLIR") {
        let path = format!("{dir}/{name}");
        let _ = fs::create_dir_all(&path);
        Some(path.into())
    } else {
        None
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_scope(scope: &cubecl_core::prelude::Scope, name: &str) {
    if let Some(path) = get_dump_name(name) {
        std::fs::write(path.join("cubecl.ir.txt"), format!("{}", scope)).unwrap();
    }
}

#[cfg(feature = "mlir-dump")]
fn dump_opt(opt: &cubecl_opt::Optimizer, name: &str) {
    if let Some(path) = get_dump_name(name) {
        std::fs::write(path.join("cubecl-opt.ir.txt"), format!("{}", opt)).unwrap();
        std::fs::write(path.join("cubecl-opt.ir.dot"), opt.main.dot_viz()).unwrap();
    }
}

pub mod dialect;
pub mod jit;

#[cfg(feature = "pliron-dump")]
use std::{path::PathBuf, str::FromStr};

use cubecl_common::backtrace::BackTrace;
use cubecl_opt::passes::simple_cse::SimpleCSEPass;
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{
    Compiler,
    ir::rewrite::SimplifyOpsPass,
    post_processing::{bitwise::PromoteBitwisePass, disaggregate::DisaggregatePass},
    prelude::*,
};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    op::Op,
    opts::{constants::sccp::SCCPPass, dce::DCEPass},
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
};

use crate::compiler::dialect::cpu::InsertConstantEmulationPass;
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
        let module = kernel.body.state().module;
        let module_op = module.get_operation();
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        #[cfg(not(feature = "pliron-dump"))]
        let ir_printing_dir = None;
        #[cfg(feature = "pliron-dump")]
        let ir_printing_dir = pliron_path(&kernel.settings.kernel_name);
        let config = PMConfig {
            print_before_all: true,
            print_after_all: true,
            ir_printing_dir,
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(InsertConstantEmulationPass);
        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        PlironEngine
    }
}

#[cfg(feature = "pliron-dump")]
fn pliron_path(name: &str) -> Option<PathBuf> {
    use std::fs;
    if let Ok(dir) = std::env::var("CUBECL_DEBUG_PLIRON") {
        let path = PathBuf::from_str(&dir).unwrap().join(name);
        let _ = fs::create_dir_all(&path);
        Some(path)
    } else {
        None
    }
}

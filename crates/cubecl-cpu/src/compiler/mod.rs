pub mod dialect;
pub mod jit;

#[cfg(feature = "pliron-dump")]
use pliron::context::Context;
use pliron_llvm::builtin_to_llvm::builtin_to_llvm_pass;
// use pliron_llvm::builtin_to_llvm::builtin_to_llvm_pass;
#[cfg(feature = "pliron-dump")]
use std::{path::PathBuf, str::FromStr};

use cubecl_common::backtrace::BackTrace;
use cubecl_opt::passes::simple_cse::SimpleCSEPass;
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{
    ir::rewrite::SimplifyOpsPass,
    post_processing::{bitwise::PromoteBitwisePass, disaggregate::DisaggregatePass},
    prelude::*,
    Compiler,
};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    op::Op,
    operation::verify_operation,
    opts::{
        constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass,
        simplify_cfg::SimplifyCFGPass,
    },
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
};

use crate::compiler::{
    dialect::{
        branch::CfToLlvmConversionPass, entrypoint::InsertConstantEmulationPass,
        metadata::LowerEntryAbiPass, to_llvm::CubeToLLVMPass,
    },
    jit::engine::PlironEngine,
};

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
        func_passes.add_pass(CfToLlvmConversionPass::default());
        func_passes.add_pass(SimplifyCFGPass);
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(LowerEntryAbiPass::default());
        func_passes.add_pass(CubeToLLVMPass::default());
        func_passes.add_pass(Mem2RegPass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(builtin_to_llvm_pass());

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        verify_operation(module_op, &ctx).expect("Failed to verify after control-flow lowering");

        PlironEngine::compile(&ctx, module, &kernel.settings.kernel_name)
            .expect("Failed to convert to LLVM IR")
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

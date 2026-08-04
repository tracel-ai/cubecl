use super::shader::ComputeShader;
use crate::compiler::wgsl::{
    self, EnableFeaturesPass, builtin::LowerBuiltinsPass, lower::LowerOpsWgslPass,
    metadata::declare_info, rewrite_args, shared_memory_size,
};

use cubecl_core::{
    WgpuCompilationOptions,
    backtrace::BackTrace,
    post_processing::{
        checked_io::{CheckedIo, CheckedIoPass},
        disaggregate::DisaggregatePass,
        expression_merge::RemoveTrivialOpsPass,
        saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
};
use cubecl_ir::{
    ContextExt,
    pliron::{
        builtin::ops::{FuncOp, ModuleOp},
        operation::verify_operation,
        opts::{constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass},
    },
    prelude::{AnalysisManager, NestedOpsPass, Op, OpPass, PMConfig, Pass, Passes},
    rewrite::SimplifyOpsPass,
    settings::Dim3,
};
use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, simple_cse::SimpleCSEPass,
};
use cubecl_runtime::compiler::CompilationError;
use cubecl_runtime::kernel;

const MAX_VECTOR_SIZE: usize = 4;

pub struct KernelInfo {
    pub cube_dim: Dim3,
}

/// Wgsl Compiler.
#[derive(Clone, Default)]
pub struct WgslCompiler;

impl core::fmt::Debug for WgslCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgslCompiler")
    }
}

impl cubecl_core::Compiler for WgslCompiler {
    type Representation = ComputeShader;
    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        shader: kernel::KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        self.compile_shader(shader, compilation_options)
    }

    fn extension(&self) -> &'static str {
        "wgsl"
    }
}

impl WgslCompiler {
    fn compile_shader(
        &mut self,
        value: kernel::KernelDefinition,
        compilation_options: &WgpuCompilationOptions,
    ) -> Result<wgsl::ComputeShader, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile wgsl kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        #[cfg(feature = "pliron-dump")]
        let ir_printing_dir = kernel_dir_name(&value.settings.kernel_name);

        let module = value.body.state().module;
        let entry_func = value.body.state().entry_func;
        let module_op = module.get_operation();
        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty(value.info);
        ctx.set_aux_ty(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });
        ctx.set_aux_ty(*compilation_options);

        #[cfg(feature = "pliron-dump")]
        if let Some(print_dir) = &ir_printing_dir {
            use pliron::printable::Printable;
            let str = std::format!("{}", module_op.disp(&ctx));
            std::fs::write(print_dir.join("initial.plir"), &str).unwrap();
        }

        verify_operation(module_op, &ctx)?;

        let config = PMConfig {
            #[cfg(feature = "pliron-dump")]
            ir_printing_dir,
            print_after_all: cfg!(feature = "pliron-dump"),
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(CheckedIoPass::new(CheckedIo::new(
            value.settings.execution_mode,
            value.settings.kernel_name.clone(),
        )));
        func_passes.add_pass(UnrollPass::new(MAX_VECTOR_SIZE));

        func_passes.add_pass(LowerOpsWgslPass::default());
        func_passes.add_pass(LowerSaturatingArithmeticPass::default());
        func_passes.add_pass(LowerBuiltinsPass);

        func_passes.add_pass(RemoveTrivialOpsPass::default());

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);

        // SCCP/DCE may unlock more mem2reg opportunities, and vice versa. So we do a sandwich.
        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);
        passes.add_pass(EnableFeaturesPass);

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        let buffers = rewrite_args(&mut ctx, entry_func);
        declare_info(&mut ctx, entry_func, buffers.len());
        let shared_memory_size = shared_memory_size(&ctx, module_op);

        verify_operation(module.get_operation(), &ctx)?;

        Ok(ComputeShader {
            buffers,
            shared_memory_size,
            ctx,
        })
    }
}

#[cfg(feature = "pliron-dump")]
pub fn kernel_dir_name(name: &str) -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("CUBECL_DEBUG_PLIRON") {
        let path = sanitize_filename::sanitize_with_options(
            name,
            sanitize_filename::Options {
                replacement: "_",
                ..Default::default()
            },
        );
        let dir = std::path::PathBuf::from(dir).join(&path);
        std::fs::create_dir_all(&dir).unwrap();
        Some(dir)
    } else {
        None
    }
}

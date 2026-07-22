use super::shader::ComputeShader;
use crate::compiler::wgsl::{
    self, EnableFeaturesPass, builtin::LowerBuiltinsPass, lower::LowerOpsWgslPass,
    metadata::declare_info, rewrite_args, shared_memory_size,
};

use cubecl_core::{
    WgpuCompilationOptions,
    backtrace::BackTrace,
    post_processing::{
        disaggregate::DisaggregatePass, saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
};
use cubecl_ir::{
    ContextExt,
    pliron::{
        builtin::ops::{FuncOp, ModuleOp},
        operation::verify_operation,
        opts::{constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass},
        printable::Printable,
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

        let module = value.body.state().module;
        let entry_func = value.body.state().entry_func;
        let module_op = module.get_operation();
        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty(value.info);
        ctx.set_aux_ty(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });
        ctx.set_aux_ty(*compilation_options);

        std::fs::write("target/initial.plir", format!("{}", module.disp(&ctx))).unwrap();
        verify_operation(module_op, &ctx).expect("Failed to verify before passes");

        let config = PMConfig {
            print_after_all: true,
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        // func_passes.add_pass(LowerInfoPass);
        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(UnrollPass::new(MAX_VECTOR_SIZE));

        func_passes.add_pass(LowerOpsWgslPass::default());

        func_passes.add_pass(LowerSaturatingArithmeticPass::default());

        func_passes.add_pass(LowerBuiltinsPass);

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

        std::fs::write(
            "target/after_lower_shared.plir",
            format!("{}", module.disp(&ctx)),
        )
        .unwrap();

        let buffers = rewrite_args(&mut ctx, entry_func);
        declare_info(&mut ctx, entry_func, buffers.len());
        let shared_memory_size = shared_memory_size(&ctx, module_op);

        std::fs::write(
            "target/after_convert_args.plir",
            format!("{}", module.disp(&ctx)),
        )
        .unwrap();

        verify_operation(module.get_operation(), &ctx).expect("Failed to verify after passes");

        Ok(ComputeShader {
            buffers,
            shared_memory_size,
            ctx,
        })
    }

    // fn compile_operation(
    //     &mut self,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     operation: cube::Operation,
    //     out: Option<cube::ExpandValue>,
    //     scope: &cube::Scope,
    // ) {
    //     match operation {
    //         cube::Operation::Atomic(op) => instructions.push(self.compile_atomic(op, out)),
    //         cube::Operation::Synchronization(val) => {
    //             self.compile_synchronization(instructions, val)
    //         }
    //     }
    // }

    // fn compile_synchronization(
    //     &mut self,
    //     instructions: &mut Vec<wgsl::Instruction>,
    //     synchronization: cube::Synchronization,
    // ) {
    //     match synchronization {
    //         cube::Synchronization::SyncCube => {
    //             instructions.push(wgsl::Instruction::WorkgroupBarrier)
    //         }
    //         cube::Synchronization::SyncPlane => {
    //             panic!("Synchronization within a plane is not supported in WGSL")
    //         }
    //         cube::Synchronization::SyncStorage => {
    //             instructions.push(wgsl::Instruction::StorageBarrier)
    //         }
    //         cube::Synchronization::SyncAsyncProxyShared => panic!("TMA is not supported in WGSL"),
    //     };
    // }

    // fn compile_comment(&mut self, instructions: &mut Vec<wgsl::Instruction>, content: String) {
    //     instructions.push(wgsl::Instruction::Comment { content })
    // }

    // fn compile_atomic(
    //     &mut self,
    //     atomic: cube::AtomicOp,
    //     out: Option<cube::ExpandValue>,
    // ) -> wgsl::Instruction {
    //     match atomic {
    //         cube::AtomicOp::Add(op) => wgsl::Instruction::AtomicAdd {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Sub(op) => wgsl::Instruction::AtomicSub {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Max(op) => wgsl::Instruction::AtomicMax {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Min(op) => wgsl::Instruction::AtomicMin {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::And(op) => wgsl::Instruction::AtomicAnd {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Or(op) => wgsl::Instruction::AtomicOr {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Xor(op) => wgsl::Instruction::AtomicXor {
    //             ptr: self.compile_value(op.ptr),
    //             value: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Load(ptr) => wgsl::Instruction::AtomicLoad {
    //             input: self.compile_value(ptr),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::Store(op) => wgsl::Instruction::AtomicStore {
    //             input: self.compile_value(op.value),
    //             out: self.compile_value(op.ptr),
    //         },
    //         cube::AtomicOp::Swap(op) => wgsl::Instruction::AtomicSwap {
    //             lhs: self.compile_value(op.ptr),
    //             rhs: self.compile_value(op.value),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //         cube::AtomicOp::CompareAndSwap(op) => wgsl::Instruction::AtomicCompareExchangeWeak {
    //             ptr: self.compile_value(op.ptr),
    //             cmp: self.compile_value(op.cmp),
    //             value: self.compile_value(op.val),
    //             out: self.compile_value(out.unwrap()),
    //         },
    //     }
    // }
}

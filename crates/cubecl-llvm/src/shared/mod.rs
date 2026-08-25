pub mod branch;
pub mod entrypoint;
pub mod jit;
pub mod metadata;
pub mod polyfill;
pub mod shared_memory;
pub mod to_llvm;

use core::cell::RefCell;
use cubecl_runtime::kernel::BufferIOAttr;
use std::rc::Rc;

use cubecl_environment::backtrace::BackTrace;
use pliron_llvm::builtin_to_llvm::builtin_to_llvm_pass;
#[cfg(feature = "pliron-dump")]
use std::{path::PathBuf, str::FromStr};

use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, inst_combine::InstCombinePass,
    simple_cse::SimpleCSEPass, sroa::SROAPass,
};
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{
    Compiler,
    ir::dialect::scf::BranchToSCFPass,
    ir::rewrite::SimplifyOpsPass,
    post_processing::bitwise::PromoteBitwisePass,
    post_processing::minifloat::{LowerMinifloatCastPass, LowerMinifloatComparePass},
    prelude::*,
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
    printable::Printable,
};

use crate::amdgpu::abi::KernargArgs;
use crate::amdgpu::builtins::InsertAmdgpuBuiltinsPass;
use crate::amdgpu::plane_dim_for;
use crate::shared::{
    branch::SCFToLlvmCf,
    entrypoint::InsertConstantEmulationPass,
    jit::engine::{KernelRequirements, PlironEngine},
    metadata::{LowerEntryAbiPass, TableArgs},
    polyfill::{LowerComplexOpPass, synchronization::uses_cube_barrier},
    shared_memory::SharedMemories,
    shared_memory::declares_shared_memory,
    to_llvm::CubeToLLVMPass,
};

/// Which machine the pliron pipeline is lowering for.
///
/// The target is chosen once, from the environment, before any device is known.
/// The specific gfx architecture is *not* part of it: that is a property of the
/// device being compiled for and arrives later, on [`PlironOptions`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LlvmTarget {
    #[default]
    Cpu,
    AmdGpu,
}

#[derive(Clone, Debug, Default)]
pub struct PlironCompiler {
    pub target: LlvmTarget,
}

#[derive(Clone, Debug, Default)]
pub struct PlironOptions {
    /// gfx name of the target device, e.g. `"gfx1201"`. Ignored by [`LlvmTarget::Cpu`].
    pub arch: String,
}

/// A finished AMD code object, compiled and linked by this crate.
#[derive(Clone, Debug)]
pub struct AmdGpuModule {
    /// A linked `ET_DYN` code object, ready for `hipModuleLoadData`.
    pub code_object: Vec<u8>,
    /// Symbol name of the `amdgpu_kernel` entry point.
    pub entrypoint: String,
    /// Textual IR, kept for logging and for hashing into the compilation cache.
    pub ir: String,
    /// AMDGPU assembly, populated only when `CUBECL_DEBUG_PLIRON` is set.
    pub asm: Option<String>,
}

/// What [`PlironCompiler`] produces. Both targets yield something directly
/// runnable: the CPU a JIT'd function, the GPU a linked code object.
#[derive(Clone)]
pub enum PlironArtifact {
    Jit(PlironEngine),
    AmdGpuCode(AmdGpuModule),
}

impl PlironArtifact {
    /// The JIT engine, for hosts that only ever compile for the CPU.
    pub fn expect_jit(self) -> PlironEngine {
        match self {
            PlironArtifact::Jit(engine) => engine,
            PlironArtifact::AmdGpuCode(_) => {
                panic!("expected a JIT engine, got an AMDGPU code object")
            }
        }
    }
}

impl core::fmt::Display for PlironArtifact {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlironArtifact::Jit(engine) => write!(f, "{engine}"),
            PlironArtifact::AmdGpuCode(module) => write!(f, "{}", module.ir),
        }
    }
}

impl Compiler for PlironCompiler {
    type Representation = PlironArtifact;

    type CompilationOptions = PlironOptions;

    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        match repr {
            PlironArtifact::Jit(engine) => Some(engine.buffer_io().to_vec()),
            // The AMDGPU pipeline does not run `AnnotateGlobalVisibilityPass`, so
            // there is no stamped answer to report; `None` is the conservative
            // reading of every buffer as both read and written.
            PlironArtifact::AmdGpuCode(_) => None,
        }
    }

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
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

        Ok(self.clone().compile_ir(kernel, compilation_options))
    }

    fn extension(&self) -> &'static str {
        match self.target {
            LlvmTarget::Cpu => "plir",
            LlvmTarget::AmdGpu => "ll",
        }
    }
}

impl PlironCompiler {
    fn compile_ir(self, kernel: KernelDefinition, options: &PlironOptions) -> PlironArtifact {
        match self.target {
            LlvmTarget::Cpu => PlironArtifact::Jit(self.compile_cpu(kernel)),
            LlvmTarget::AmdGpu => {
                PlironArtifact::AmdGpuCode(self.compile_amdgpu(kernel, &options.arch))
            }
        }
    }

    fn compile_cpu(self, kernel: KernelDefinition) -> PlironEngine {
        let module = kernel.body.state().module;
        let entry_func = kernel.body.state().entry_func;
        let module_op = module.get_operation();
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        let needs_parallelism = kernel.settings.cube_dim.num_elems() > 1
            && (uses_cube_barrier(&ctx, module_op) || declares_shared_memory(&ctx, module_op));
        // Filled in by the entry ABI pass, which is where the shared memories get their slot.
        let shared_memories = Rc::new(RefCell::new(SharedMemories::default()));

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
        func_passes.add_pass(SROAPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(InstCombinePass::default());
        func_passes.add_pass(LowerMinifloatCastPass::default());
        func_passes.add_pass(LowerMinifloatComparePass::default());
        func_passes.add_pass(LowerComplexOpPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SROAPass);

        let mut lowering_passes = OpPass::<FuncOp, Passes>::default();
        lowering_passes.add_pass(BranchToSCFPass::default());
        lowering_passes.add_pass(SCFToLlvmCf::default());
        lowering_passes.add_pass(LowerEntryAbiPass::new(
            kernel.info.clone(),
            Box::new(TableArgs::new(shared_memories.clone())),
        ));
        lowering_passes.add_pass(CubeToLLVMPass::default());
        lowering_passes.add_pass(SimplifyCFGPass);
        lowering_passes.add_pass(DCEPass);
        lowering_passes.add_pass(Mem2RegPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        // Reads cube-dialect memory effects, so it has to run after the
        // optimizations that shape them and before the lowering group erases
        // the cube ops — the same post-optimization point the other backends
        // annotate at.
        passes.add_pass(AnnotateGlobalVisibilityPass);
        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        // Read the stamped answer now: the entry ABI lowering below folds the
        // buffer arguments behind a pointer table and erases them, attributes
        // included.
        let io = cubecl_core::ir::attributes::buffer_io_by_position(&ctx, entry_func)
            .into_iter()
            .collect();

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        passes.add_pass(NestedOpsPass::new(lowering_passes));
        passes.add_pass(builtin_to_llvm_pass());
        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        if let Err(e) = verify_operation(module_op, &ctx) {
            panic!("{}", e.disp(&ctx));
        }

        let requirements = KernelRequirements {
            needs_parallelism,
            shared_memories: shared_memories.take(),
        };

        PlironEngine::compile(&ctx, module, &kernel.settings.kernel_name, requirements, io)
            .expect("Failed to convert to LLVM IR")
    }

    /// Lowers `kernel` for `arch` and compiles it into a linked AMD code object.
    ///
    /// Mirrors [`Self::compile_cpu`] with three substitutions: the builtins come
    /// from the hardware rather than an emulated loop nest, the entry ABI is
    /// kernargs rather than an indirection table, and the backend is a
    /// `TargetMachine` rather than the JIT.
    fn compile_amdgpu(self, kernel: KernelDefinition, arch: &str) -> AmdGpuModule {
        let module = kernel.body.state().module;
        let module_op = module.get_operation();
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        // Checked before any pass runs: both of these would otherwise be lowered
        // by passes that assume the CPU host, and miscompile silently.
        if uses_cube_barrier(&ctx, module_op) {
            unimplemented!("cube barriers are not supported on the AMDGPU target yet");
        }
        if declares_shared_memory(&ctx, module_op) {
            unimplemented!("shared memory is not supported on the AMDGPU target yet");
        }

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
        func_passes.add_pass(InsertAmdgpuBuiltinsPass {
            plane_dim: plane_dim_for(arch),
        });
        func_passes.add_pass(SROAPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(LowerMinifloatCastPass::default());
        func_passes.add_pass(LowerMinifloatComparePass::default());
        func_passes.add_pass(LowerComplexOpPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SROAPass);
        func_passes.add_pass(BranchToSCFPass::default());
        func_passes.add_pass(SCFToLlvmCf::default());
        func_passes.add_pass(LowerEntryAbiPass::new(
            kernel.info.clone(),
            Box::new(KernargArgs),
        ));
        func_passes.add_pass(CubeToLLVMPass::default());
        func_passes.add_pass(SimplifyCFGPass);
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(Mem2RegPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(builtin_to_llvm_pass());

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        if let Err(e) = verify_operation(module_op, &ctx) {
            panic!("{}", e.disp(&ctx));
        }

        crate::amdgpu::codegen::emit_code_object(
            &ctx,
            module,
            &kernel.settings.kernel_name,
            arch,
            kernel.settings.cube_dim.num_elems(),
        )
        .unwrap_or_else(|err| {
            panic!(
                "Failed to compile '{}' for {arch}: {err}",
                kernel.settings.kernel_name
            )
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_target_is_cpu() {
        assert!(matches!(PlironCompiler::default().target, LlvmTarget::Cpu));
    }

    #[test]
    fn amdgpu_artifact_displays_its_ir() {
        let artifact = PlironArtifact::AmdGpuCode(AmdGpuModule {
            code_object: vec![0x7f, 0x45, 0x4c, 0x46],
            entrypoint: "k".to_string(),
            ir: "define void @k() { ret void }".to_string(),
            asm: None,
        });
        assert!(artifact.to_string().contains("@k"));
    }
}

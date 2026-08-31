use core::cell::RefCell;
use cubecl_runtime::kernel::BufferIOAttr;
use std::rc::Rc;

use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::Bytes;
use pliron_llvm::builtin_to_llvm::builtin_to_llvm_pass;
#[cfg(feature = "pliron-dump")]
use std::{path::PathBuf, str::FromStr};

use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, inst_combine::InstCombinePass,
    sccp::SCCPPass, simple_cse::SimpleCSEPass, sroa::SROAPass,
};
use cubecl_runtime::compiler::CompilationError;

use cubecl_core::{
    Compiler,
    ir::amd::GfxArch,
    ir::dialect::scf::BranchToSCFPass,
    ir::metadata::Info,
    ir::rewrite::SimplifyOpsPass,
    post_processing::bitwise::PromoteBitwisePass,
    post_processing::minifloat::{LowerMinifloatCastPass, LowerMinifloatComparePass},
    prelude::*,
};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    operation::verify_operation,
    opts::{dce::DCEPass, mem2reg::Mem2RegPass, simplify_cfg::SimplifyCFGPass},
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
    printable::Printable,
};

use crate::amdgpu::abi::AmdGpuLowering;
use crate::amdgpu::matrix::CtxWmma;
use crate::amdgpu::plane::CtxPlaneDim;
use crate::amdgpu::shared_memory::CtxSharedMemory;
use crate::cpu::{
    abi::CpuLowering,
    jit::engine::{KernelRequirements, PlironEngine},
    shared_memory::SharedMemories,
    synchronization::uses_cube_barrier,
};
use crate::shared::{
    branch::SCFToLlvmCf, lowering::TargetLowering, metadata::LowerEntryAbiPass,
    polyfill::LowerComplexOpPass, shared_memory::declares_shared_memory, to_llvm::CubeToLLVMPass,
};
use crate::target::{CtxTarget, LlvmTarget};

#[derive(Clone, Debug, Default)]
pub struct PlironCompiler {
    pub target: LlvmTarget,
}

#[derive(Clone, Debug, Default)]
pub struct PlironOptions {
    /// The device [`LlvmTarget::AmdGpu`] compiles for. `None` on the CPU, which has no gfx
    /// architecture to name.
    pub arch: Option<GfxArch>,
}

/// A finished AMD code object, compiled and linked by this crate.
#[derive(Clone, Debug)]
pub struct AmdGpuModule {
    /// A linked `ET_DYN` code object, ready for `hipModuleLoadData`.
    pub code_object: Bytes,
    /// Symbol name of the `amdgpu_kernel` entry point.
    pub entrypoint: String,
    /// Textual IR, kept for logging and for hashing into the compilation cache.
    pub ir: String,
    /// AMDGPU assembly, populated only when `CUBECL_DEBUG_PLIRON` is set.
    pub asm: Option<String>,
    /// Bytes of LDS a launch must reserve, which the kernel takes as dynamic shared memory.
    pub shared_memory_size: usize,
    /// What the kernel does with each buffer binding, by buffer position, as
    /// stamped by `AnnotateGlobalVisibilityPass` before the entry ABI lowering
    /// folded the buffer arguments away.
    pub io: Vec<BufferIOAttr>,
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
            PlironArtifact::AmdGpuCode(module) => Some(module.io.clone()),
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

        self.clone().compile_ir(kernel, compilation_options)
    }

    fn extension(&self) -> &'static str {
        match self.target {
            LlvmTarget::Cpu => "plir",
            LlvmTarget::AmdGpu => "ll",
        }
    }
}

impl PlironCompiler {
    fn compile_ir(
        self,
        kernel: KernelDefinition,
        options: &PlironOptions,
    ) -> Result<PlironArtifact, CompilationError> {
        match self.target {
            LlvmTarget::Cpu => Ok(PlironArtifact::Jit(self.compile_cpu(kernel)?)),
            LlvmTarget::AmdGpu => {
                let arch = options.arch.as_ref().ok_or_else(|| {
                    generic("the AMDGPU target needs the device it compiles for".to_string())
                })?;
                Ok(PlironArtifact::AmdGpuCode(
                    self.compile_amdgpu(kernel, arch)?,
                ))
            }
        }
    }

    fn compile_cpu(self, kernel: KernelDefinition) -> Result<PlironEngine, CompilationError> {
        let module = kernel.body.state().module;
        let module_op = module.get_operation();
        let ir = KernelIr::of(&kernel);
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        ctx.set_target(LlvmTarget::Cpu);

        let needs_parallelism = kernel.settings.cube_dim.num_elems() > 1
            && (uses_cube_barrier(&ctx, module_op) || declares_shared_memory(&ctx, module_op));
        // Filled in by the entry ABI pass, which is where the shared memories get their slot.
        let shared_memories = Rc::new(RefCell::new(SharedMemories::default()));

        let io = lower(&mut ctx, &ir, &CpuLowering::new(shared_memories.clone()))?;

        let requirements = KernelRequirements {
            needs_parallelism,
            shared_memories: shared_memories.take(),
        };

        PlironEngine::compile(&ctx, module, &kernel.settings.kernel_name, requirements, io)
            .map_err(|err| generic(format!("converting to LLVM IR: {err}")))
    }

    /// Lowers `kernel` for `arch` and compiles it into a linked AMD code object.
    fn compile_amdgpu(
        self,
        kernel: KernelDefinition,
        arch: &GfxArch,
    ) -> Result<AmdGpuModule, CompilationError> {
        let module = kernel.body.state().module;
        let ir = KernelIr::of(&kernel);
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        // The runtime checks this against what the driver reports before it compiles
        // anything, so an architecture with no known width should not reach here.
        let plane_dim = arch.plane_dim().ok_or_else(|| {
            generic(format!(
                "no known wavefront width for '{}', so a kernel cannot be generated for it",
                arch.name()
            ))
        })?;

        ctx.set_target(LlvmTarget::AmdGpu);
        // Left at zero for the kernels that never declare any.
        ctx.set_shared_memory_size(0);
        ctx.set_plane_dim(plane_dim);
        ctx.set_wmma(arch.wmma());

        let io = lower(&mut ctx, &ir, &AmdGpuLowering { plane_dim })?;

        // Filled in by the block's lowering, which is the last point it is known.
        let shared_memory_size = ctx.shared_memory_size();

        crate::amdgpu::codegen::emit_code_object(
            &ctx,
            module,
            &kernel.settings.kernel_name,
            arch,
            kernel.settings.cube_dim.num_elems(),
            shared_memory_size,
            io,
        )
        .map_err(|err| {
            generic(format!(
                "compiling '{}' for {}: {err}",
                kernel.settings.kernel_name,
                arch.name()
            ))
        })
    }
}

/// The module and the facts about it the pipeline needs, taken while the kernel still owns
/// its scope: lowering starts by consuming that scope into the context, after which the
/// kernel cannot be asked anything.
struct KernelIr {
    module_op: Ptr<Operation>,
    entry_func: FuncOp,
    info: Info,
    #[cfg_attr(not(feature = "pliron-dump"), allow(dead_code))]
    name: String,
}

impl KernelIr {
    fn of(kernel: &KernelDefinition) -> Self {
        let state = kernel.body.state();
        Self {
            module_op: state.module.get_operation(),
            entry_func: state.entry_func,
            info: kernel.info.clone(),
            name: kernel.settings.kernel_name.clone(),
        }
    }
}

/// Runs `kernel` down to the LLVM dialect, with `target` contributing the passes around the
/// optimizations both share, and answers what the kernel does with each buffer binding.
///
/// The context is left holding a verified LLVM-dialect module, which each target then takes
/// to machine code its own way.
fn lower(
    ctx: &mut Context,
    kernel: &KernelIr,
    target: &dyn TargetLowering,
) -> Result<Vec<BufferIOAttr>, CompilationError> {
    let (module_op, entry_func) = (kernel.module_op, kernel.entry_func);

    #[cfg(not(feature = "pliron-dump"))]
    let ir_printing_dir = None;
    #[cfg(feature = "pliron-dump")]
    let ir_printing_dir = pliron_path(&kernel.name);
    let config = PMConfig {
        print_after_all: true,
        ir_printing_dir,
        ..Default::default()
    };

    let mut analyses = AnalysisManager::default();
    analyses.set_config(config);

    let mut func_passes = OpPass::<FuncOp, Passes>::default();
    target.prologue(&mut func_passes);
    func_passes.add_pass(SROAPass);
    func_passes.add_pass(SCCPPass);
    func_passes.add_pass(SimpleCSEPass);
    func_passes.add_pass(SimplifyOpsPass::default());
    func_passes.add_pass(PromoteBitwisePass);
    func_passes.add_pass(InstCombinePass::default());
    func_passes.add_pass(LowerMinifloatCastPass::default());
    func_passes.add_pass(LowerMinifloatComparePass::default());
    func_passes.add_pass(LowerComplexOpPass::default());
    target.epilogue(&mut func_passes);
    func_passes.add_pass(DCEPass);
    func_passes.add_pass(SROAPass);

    let mut lowering_passes = OpPass::<FuncOp, Passes>::default();
    lowering_passes.add_pass(BranchToSCFPass::default());
    lowering_passes.add_pass(SCFToLlvmCf::default());
    lowering_passes.add_pass(LowerEntryAbiPass::new(
        kernel.info.clone(),
        target.arg_layout(),
    ));
    lowering_passes.add_pass(CubeToLLVMPass::default());
    lowering_passes.add_pass(SimplifyCFGPass);
    lowering_passes.add_pass(DCEPass);
    lowering_passes.add_pass(Mem2RegPass);

    let mut passes = OpPass::<ModuleOp, Passes>::default();
    passes.add_pass(NestedOpsPass::new(func_passes));
    // Reads cube-dialect memory effects, so it has to run after the optimizations that shape
    // them and before the lowering group erases the cube ops, which is the same
    // post-optimization point the other backends annotate at.
    passes.add_pass(AnnotateGlobalVisibilityPass);
    run(&mut passes, module_op, ctx, &mut analyses)?;

    // Read the stamped answer now: the entry ABI lowering below folds the buffer arguments
    // behind the target's own layout and erases them, attributes included.
    let io = cubecl_core::ir::attributes::buffer_io_by_position(ctx, entry_func);

    let mut passes = OpPass::<ModuleOp, Passes>::default();
    passes.add_pass(NestedOpsPass::new(lowering_passes));
    passes.add_pass(builtin_to_llvm_pass());
    run(&mut passes, module_op, ctx, &mut analyses)?;

    verify_operation(module_op, ctx).map_err(|err| {
        generic(format!(
            "the lowered module does not verify: {}",
            err.disp(ctx)
        ))
    })?;

    Ok(io)
}

/// Runs `passes`, reporting a failure as a compilation error rather than unwinding.
fn run(
    passes: &mut OpPass<ModuleOp, Passes>,
    module_op: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
) -> Result<(), CompilationError> {
    passes
        .run(module_op, ctx, analyses)
        .map(|_| ())
        .map_err(|err| generic(format!("{}", err.disp(ctx))))
}

/// A compilation error carrying `reason`.
///
/// Everything below the `Compiler` impl reports with a `String`, and this is where those
/// become the error the server enqueues: a kernel the backend cannot compile is a failed
/// launch to report, not a process to take down.
fn generic(reason: String) -> CompilationError {
    CompilationError::Generic {
        reason,
        backtrace: BackTrace::capture(),
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
            code_object: Bytes::from_bytes_vec(vec![0x7f, 0x45, 0x4c, 0x46]),
            entrypoint: "k".to_string(),
            ir: "define void @k() { ret void }".to_string(),
            asm: None,
            shared_memory_size: 0,
            io: Vec::new(),
        });
        assert!(artifact.to_string().contains("@k"));
    }
}

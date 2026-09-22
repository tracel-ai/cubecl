#[cfg(feature = "amdgpu")]
use crate::amdgpu::{abi::AmdGpuLowering, matrix::CtxWmma};
#[cfg(feature = "nvptx")]
use crate::nvptx::{
    abi::NvptxLowering,
    codegen::{MetadataParams, NvptxEntry},
    ptx_version::PtxVersion,
};
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
use crate::shared::{plane::CtxPlaneDim, shared_memory::CtxSharedMemory};
use crate::{
    cpu::{
        abi::CpuLowering,
        jit::engine::{KernelRequirements, PlironEngine},
        shared_memory::SharedMemories,
        synchronization::uses_cube_barrier,
    },
    prelude::{
        AnalysisManager, Context, ContextExt, CtxTarget, FuncOp, LlvmTarget, ModuleOp,
        NestedOpsPass, Op, OpPass, Operation, PMConfig, Pass, Passes, Printable, Ptr,
        TargetLowering,
    },
    shared::{
        branch::SCFToLlvmCf,
        metadata::{CtxGridConstants, LowerEntryAbiPass},
        polyfill::LowerComplexOpPass,
        shared_memory::declares_shared_memory,
        to_llvm::CubeToLLVMPass,
    },
};
use core::cell::RefCell;
#[cfg(feature = "nvptx")]
use cubecl_core::ir::nvidia::SmArch;
use cubecl_core::{
    Compiler,
    ir::{amd::GfxArch, dialect::scf::BranchToSCFPass, metadata::Info, rewrite::SimplifyOpsPass},
    post_processing::{
        bitwise::PromoteBitwisePass,
        minifloat::{LowerMinifloatCastPass, LowerMinifloatComparePass},
    },
    prelude::*,
};
use cubecl_environment::backtrace::BackTrace;
#[cfg(feature = "amdgpu")]
use cubecl_environment::bytes::Bytes;
use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, inst_combine::InstCombinePass,
    sccp::SCCPPass, simple_cse::SimpleCSEPass, sroa::SROAPass,
};
use cubecl_runtime::{
    compiler::CompilationError, config::compilation::F16Evaluation, kernel::BufferIOAttr,
};
use pliron::{
    operation::verify_operation,
    opts::{dce::DCEPass, mem2reg::Mem2RegPass, simplify_cfg::SimplifyCFGPass},
};
use pliron_llvm::builtin_to_llvm::builtin_to_llvm_pass;
use std::rc::Rc;
#[cfg(feature = "pliron-dump")]
use std::{path::PathBuf, str::FromStr};

#[derive(Clone, Debug, Default)]
pub struct PlironCompiler {
    pub target: LlvmTarget,
}

#[derive(Clone, Debug, Default)]
pub struct PlironOptions {
    /// Minimum CPU buffer alignment, including view offsets. Must be a power of two.
    /// `None` makes no alignment promise.
    pub cpu_buffer_alignment: Option<u32>,
    /// AMDGPU architecture, or `None` for other targets.
    pub arch: Option<GfxArch>,
    /// CPU f16 evaluation precision.
    pub f16_evaluation: F16Evaluation,
    /// NVPTX architecture, or `None` for other targets.
    #[cfg(feature = "nvptx")]
    pub sm_arch: Option<SmArch>,
    /// PTX version to emit, or `None` for the oldest the architecture accepts.
    #[cfg(feature = "nvptx")]
    pub ptx_version: Option<PtxVersion>,
    /// Pass scalars and static metadata in the kernel parameter block.
    /// The host must use the same layout.
    pub grid_constants: bool,
}

#[cfg(feature = "amdgpu")]
/// Compiled AMDGPU module.
#[derive(Clone, Debug)]
pub struct AmdGpuModule {
    /// Loadable AMD code object.
    pub code_object: Bytes,
    /// Kernel entry point symbol.
    pub entrypoint: String,
    /// IR for logging and cache keys.
    pub ir: String,
    /// Assembly available when `CUBECL_DEBUG_PLIRON` is set.
    pub asm: Option<String>,
    /// Shared memory required per launch, in bytes.
    pub shared_memory_size: usize,
    /// Buffer access modes in binding order.
    pub io: Vec<BufferIOAttr>,
}

/// Compiled PTX module.
#[derive(Clone, Debug)]
#[cfg(feature = "nvptx")]
pub struct NvptxModule {
    /// NUL-terminated PTX assembly.
    pub ptx: Vec<core::ffi::c_char>,
    /// Kernel entry point symbol.
    pub entrypoint: String,
    /// IR for logging and cache keys.
    pub ir: String,
    /// Shared memory required per launch, in bytes.
    pub shared_memory_size: usize,
    /// Buffer access modes in binding order.
    pub io: Vec<BufferIOAttr>,
}

/// Compiled kernel artifact.
#[derive(Clone)]
pub enum PlironArtifact {
    Jit(PlironEngine),
    #[cfg(feature = "amdgpu")]
    AmdGpuCode(AmdGpuModule),
    #[cfg(feature = "nvptx")]
    NvptxCode(NvptxModule),
}

impl PlironArtifact {
    pub fn expect_jit(self) -> PlironEngine {
        match self {
            PlironArtifact::Jit(engine) => engine,
            #[cfg(feature = "amdgpu")]
            PlironArtifact::AmdGpuCode(_) => {
                panic!("expected a JIT engine, got an AMDGPU code object")
            }
            #[cfg(feature = "nvptx")]
            PlironArtifact::NvptxCode(_) => panic!("expected a JIT engine, got a PTX module"),
        }
    }
}

impl core::fmt::Display for PlironArtifact {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlironArtifact::Jit(engine) => write!(f, "{engine}"),
            #[cfg(feature = "amdgpu")]
            PlironArtifact::AmdGpuCode(module) => write!(f, "{}", module.ir),
            #[cfg(feature = "nvptx")]
            PlironArtifact::NvptxCode(module) => write!(f, "{}", module.ir),
        }
    }
}

impl Compiler for PlironCompiler {
    type Representation = PlironArtifact;

    type CompilationOptions = PlironOptions;

    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        match repr {
            PlironArtifact::Jit(engine) => Some(engine.buffer_io().to_vec()),
            #[cfg(feature = "amdgpu")]
            PlironArtifact::AmdGpuCode(module) => Some(module.io.clone()),
            #[cfg(feature = "nvptx")]
            PlironArtifact::NvptxCode(module) => Some(module.io.clone()),
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
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => "ll",
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => "ll",
        }
    }

    fn lang_tag(&self) -> &'static str {
        match self.target {
            LlvmTarget::Cpu => "mlir",
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => "llvm",
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => "llvm",
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
            LlvmTarget::Cpu => Ok(PlironArtifact::Jit(self.compile_cpu(kernel, options)?)),
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => {
                let arch = options.arch.as_ref().ok_or_else(|| {
                    generic("the AMDGPU target needs the device it compiles for".to_string())
                })?;
                Ok(PlironArtifact::AmdGpuCode(
                    self.compile_amdgpu(kernel, arch)?,
                ))
            }
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => {
                let arch = options.sm_arch.ok_or_else(|| {
                    generic("the NVPTX target needs the device it compiles for".to_string())
                })?;
                Ok(PlironArtifact::NvptxCode(self.compile_nvptx(
                    kernel,
                    arch,
                    options.ptx_version,
                    options.grid_constants,
                )?))
            }
        }
    }

    fn compile_cpu(
        self,
        kernel: KernelDefinition,
        options: &PlironOptions,
    ) -> Result<PlironEngine, CompilationError> {
        let module = kernel.body.state().module;
        let module_op = module.get_operation();
        let ir = KernelIr::of(&kernel);
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        ctx.set_target(LlvmTarget::Cpu);
        let alignment = options.cpu_buffer_alignment.unwrap_or(1);
        assert!(alignment.is_power_of_two());
        ctx.set_aux_ty(crate::target::CpuBufferAlignment(alignment));
        ctx.set_grid_constants(false);

        let needs_parallelism = kernel.settings.cube_dim.num_elems() > 1
            && (uses_cube_barrier(&ctx, module_op) || declares_shared_memory(&ctx, module_op));
        let shared_memories = Rc::new(RefCell::new(SharedMemories::default()));

        let lowering = CpuLowering::new(shared_memories.clone(), options.f16_evaluation);
        let io = lower(&mut ctx, &ir, &lowering)?;

        let requirements = KernelRequirements {
            needs_parallelism,
            shared_memories: shared_memories.take(),
        };

        PlironEngine::compile(&ctx, module, &kernel.settings.kernel_name, requirements, io)
            .map_err(|err| generic(format!("converting to LLVM IR: {err}")))
    }

    #[cfg(feature = "amdgpu")]
    fn compile_amdgpu(
        self,
        kernel: KernelDefinition,
        arch: &GfxArch,
    ) -> Result<AmdGpuModule, CompilationError> {
        let module = kernel.body.state().module;
        let ir = KernelIr::of(&kernel);
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        let plane_dim = arch.plane_dim().ok_or_else(|| {
            generic(format!(
                "no known wavefront width for '{}', so a kernel cannot be generated for it",
                arch.name()
            ))
        })?;

        ctx.set_target(LlvmTarget::AmdGpu);
        ctx.set_grid_constants(false);
        ctx.set_shared_memory_size(0);
        ctx.set_plane_dim(plane_dim);
        ctx.set_wmma(arch.wmma());

        let io = lower(&mut ctx, &ir, &AmdGpuLowering { plane_dim })?;

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

    #[cfg(feature = "nvptx")]
    fn compile_nvptx(
        self,
        kernel: KernelDefinition,
        arch: SmArch,
        ptx_version: Option<PtxVersion>,
        grid_constants: bool,
    ) -> Result<NvptxModule, CompilationError> {
        let module = kernel.body.state().module;
        let ir = KernelIr::of(&kernel);
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        ctx.set_target(LlvmTarget::Nvptx);
        ctx.set_shared_memory_size(0);
        ctx.set_grid_constants(grid_constants);
        let plane_dim = arch.plane_dim();
        ctx.set_plane_dim(plane_dim);

        let io = lower(&mut ctx, &ir, &NvptxLowering { plane_dim })?;

        let shared_memory_size = ctx.shared_memory_size();

        let metadata = if grid_constants && ir.info.has_info() {
            MetadataParams::GridConstant {
                bytes: ir.info.dynamic_meta_offset,
                dynamic_buffer: ir.info.has_dynamic_meta,
            }
        } else {
            MetadataParams::Buffer
        };

        crate::nvptx::codegen::emit_ptx(
            &ctx,
            module,
            &kernel.settings.kernel_name,
            &arch,
            ptx_version,
            NvptxEntry {
                cube_dim: kernel.settings.cube_dim.num_elems(),
                shared_memory_size,
                io,
                metadata,
            },
        )
        .map_err(|err| {
            generic(format!(
                "compiling '{}' for sm_{}: {err}",
                kernel.settings.kernel_name,
                arch.version()
            ))
        })
    }
}

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
    func_passes.add_pass(SimpleCSEPass::with_memory());
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
    // Memory effects must be annotated before cube operations are lowered.
    passes.add_pass(AnnotateGlobalVisibilityPass);
    run(&mut passes, module_op, ctx, &mut analyses)?;

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

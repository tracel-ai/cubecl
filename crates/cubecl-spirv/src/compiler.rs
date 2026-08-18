use crate::{
    CollectVerCapExtPass, ConvertArgsPass, PARAMS_NAME, SpirvKernel,
    lower::LowerOpsSpirvPass,
    ops::{
        branch::BranchToSpirvConversionPass,
        builtin::{BUILTINS_NAME, LowerBuiltinsPass},
        memory::lower_shared,
        to_spirv_dialect::ToSpirvDialectPass,
    },
    params_storage_class,
};
use cubecl_core::{
    Compiler, WgpuCompilationOptions,
    ir::{ContextExt, attributes::FuncInterface, ident, metadata::Info, rewrite::SimplifyOpsPass},
    post_processing::{
        bitwise::PromoteBitwisePass,
        checked_io::{CheckedIo, CheckedIoPass},
        minifloat::{LowerMinifloatCast, LowerMinifloatCastPass, NativeFp8},
        saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
    prelude::{KernelDefinition, Visibility},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::{
    attributes::{ATTR_BUFFER_IO, BufferIOAttr, EntrypointInterface},
    dialect::scf::BranchToSCFPass,
    prelude::{SingleBlockRegionInterface, SymbolOpInterface},
    rewrite::visit_all_ops_of_type_mut,
    settings::{Dim3, KernelSettings},
};
use cubecl_opt::passes::{
    alloc_shared_memory::AllocateSharedMemoryBlockPass,
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, mem2reg::Mem2RegPass,
    simple_cse::SimpleCSEPass, sroa::SROAPass,
};
use cubecl_runtime::compiler::CompilationError;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::OneRegionInterface,
        ops::{FuncOp, ModuleOp},
    },
    context::Context,
    identifier::Identifier,
    irbuild::{
        inserter::BlockInsertionPoint,
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    op::Op,
    operation::verify_operation,
    opts::{constants::sccp::SCCPPass, dce::DCEPass, simplify_cfg::SimplifyCFGPass},
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
};
use pliron_spirv::{
    PlironBuilder, ToSpirvOp,
    attrs::VerCapExtAttr,
    ops::{EntryPointOp, ExecutionModeOp, SpirvModuleOp},
};
use rspirv::{
    binary::Assemble,
    dr::Module,
    spirv::{
        AddressingModel, Capability, ExecutionMode, ExecutionModel, MemoryModel, StorageClass,
    },
};
use std::{fmt::Debug, sync::Arc};

pub struct KernelInfo {
    pub cube_dim: Dim3,
}

#[derive(Clone, Copy, Default)]
pub struct SpirvCompiler;

impl Compiler for SpirvCompiler {
    type Representation = SpirvKernel;
    type CompilationOptions = WgpuCompilationOptions;

    fn compile(
        &mut self,
        value: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile spirv kernel".to_string();
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

        let entry_func = value.body.state().entry_func;
        let module = value.body.state().module;

        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty::<Info>(value.info);
        ctx.set_aux_ty::<WgpuCompilationOptions>(*compilation_options);
        ctx.set_aux_ty::<KernelInfo>(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });

        let (module, bindings, shared_size) = self.compile_kernel(
            &mut ctx,
            module,
            entry_func,
            value.settings.clone(),
            #[cfg(feature = "pliron-dump")]
            ir_printing_dir,
        )?;

        let info_visibility = Visibility::Read;
        let immediate_size = match params_storage_class(&ctx, bindings.len()) {
            StorageClass::PushConstant => Some((bindings.len() + 1) * size_of::<u64>()),
            _ => None,
        };

        let kernel = SpirvKernel {
            assembled_module: module.assemble(),
            module: Some(Arc::new(module)),
            bindings,
            shared_size,
            immediate_size,
            info_visibility,
        };

        #[cfg(feature = "pliron-dump")]
        dump_spirv(&kernel, &value.settings.kernel_name);

        Ok(kernel)
    }

    fn extension(&self) -> &'static str {
        "spv"
    }
}

impl Debug for SpirvCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("spirv")
    }
}

impl SpirvCompiler {
    pub fn compile_kernel(
        &mut self,
        ctx: &mut Context,
        module: ModuleOp,
        entry_func: FuncOp,
        settings: KernelSettings,
        #[cfg(feature = "pliron-dump")] ir_printing_dir: Option<std::path::PathBuf>,
    ) -> Result<(Module, Vec<Visibility>, usize), CompilationError> {
        let entry = entry_func.get_entry_block(ctx);
        let comp_opts = ctx.aux_ty::<WgpuCompilationOptions>();
        let module_op = module.get_operation();

        #[cfg(feature = "pliron-dump")]
        if let Some(print_dir) = &ir_printing_dir {
            use pliron::printable::Printable;
            let str = std::format!("{}", module_op.disp(ctx));
            std::fs::write(print_dir.join("initial.plir"), &str).unwrap();
        }

        verify_operation(module.get_operation(), ctx)?;

        let config = PMConfig {
            print_after_all: cfg!(feature = "pliron-dump"),
            #[cfg(feature = "pliron-dump")]
            ir_printing_dir,
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(SROAPass);
        func_passes.add_pass(CheckedIoPass::new(CheckedIo::new(
            settings.execution_mode,
            settings.kernel_name,
        )));
        func_passes.add_pass(UnrollPass::new(comp_opts.vulkan.max_vector_size));
        func_passes.add_pass(AllocateSharedMemoryBlockPass);
        func_passes.add_pass(LowerSaturatingArithmeticPass::default());
        func_passes.add_pass(LowerMinifloatCastPass::new(LowerMinifloatCast::new(
            match comp_opts.vulkan.supports_float8 {
                true => NativeFp8::ALL,
                false => NativeFp8::NONE,
            },
        )));
        func_passes.add_pass(BranchToSCFPass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(LowerBuiltinsPass);

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(LowerOpsSpirvPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SROAPass);

        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SROAPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);

        passes.run(module_op, ctx, &mut analyses).unwrap();

        let bindings = (0..entry.deref(ctx).get_num_arguments()).map(|i| {
            let io = entry_func.get_arg_attr::<BufferIOAttr>(ctx, i, &ATTR_BUFFER_IO);
            match io.expect("Should have IO attr").is_writable() {
                false => Visibility::Read,
                true => Visibility::ReadWrite,
            }
        });
        let bindings: Vec<Visibility> = bindings.collect();

        verify_operation(module_op, ctx)?;

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(BranchToSpirvConversionPass::default());
        func_passes.add_pass(Mem2RegPass);
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimplifyCFGPass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.run(module_op, ctx, &mut analyses).unwrap();

        verify_operation(module_op, ctx)?;

        let spirv_module = insert_spirv_module(ctx, module);
        let spirv_module_op = spirv_module.get_operation();

        let mut passes = OpPass::<SpirvModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(DCEPass);
        func_passes.add_pass(ToSpirvDialectPass::default());

        passes.add_pass(ConvertArgsPass);
        passes.add_pass(NestedOpsPass::new(func_passes));

        // The conversion pass reports ops it must not compile (e.g.
        // cube.poison) as errors, not bugs; surface them as a compilation
        // error instead of panicking the device thread.
        passes.run(spirv_module_op, ctx, &mut analyses)?;

        let (shared_size, shared_args) = lower_shared(ctx, spirv_module);
        declare_entry_point(ctx, spirv_module, shared_args);

        // Make sure this is the last pass so it catches all ops
        OpPass::<SpirvModuleOp, CollectVerCapExtPass>::default()
            .run(spirv_module_op, ctx, &mut analyses)
            .unwrap();

        // Something weird with the validation rules, can't be bothered to debug for the MVP.
        // Try to figure this out later.
        // verify_operation(module_op, ctx).expect("Failed to verify after passes");

        let mut builder = PlironBuilder::default();
        spirv_module.to_spirv(ctx, &mut builder)?;
        let module = builder.module();

        Ok((module, bindings, shared_size))
    }
}

fn insert_spirv_module(ctx: &mut Context, module: ModuleOp) -> SpirvModuleOp {
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let comp_opts = ctx.aux_ty::<WgpuCompilationOptions>().vulkan;

    let spirv_module = SpirvModuleOp::new(
        ctx,
        ident("kernel"),
        AddressingModel::PhysicalStorageBuffer64,
        MemoryModel::Vulkan,
    );
    rewriter.inline_region(
        ctx,
        module.get_region(ctx),
        BlockInsertionPoint::AtRegionStart(spirv_module.get_region(ctx)),
    );
    let module_body = BasicBlock::new(ctx, None, vec![]);
    module_body.insert_at_front(module.get_region(ctx), ctx);
    spirv_module
        .get_operation()
        .insert_at_front(module_body, ctx);
    let vce = VerCapExtAttr::new(
        comp_opts.max_spirv_version,
        vec![Capability::Shader],
        vec![],
    );
    spirv_module.set_attr_spirv_module_vce(ctx, vce);
    spirv_module
}

fn declare_entry_point(ctx: &mut Context, module: SpirvModuleOp, shared_args: Vec<Identifier>) {
    let op = module.get_operation();
    visit_all_ops_of_type_mut::<FuncOp, _>(
        ctx,
        &mut (module, shared_args),
        op,
        |ctx, (module, shared_args), func| {
            let Some(entry) = func.get_entrypoint_abi(ctx) else {
                return;
            };
            let block = module.get_body(ctx, 0);
            let func_name = func.get_symbol_name(ctx);
            let mut interface = vec![PARAMS_NAME.clone(), BUILTINS_NAME.clone()];
            interface.extend(shared_args.clone());
            let entry_point = EntryPointOp::new(
                ctx,
                ExecutionModel::GLCompute,
                func_name.clone(),
                func_name.to_string(),
                interface,
            );
            entry_point.get_operation().insert_at_front(block, ctx);
            let (x, y, z) = entry.cube_dim.into();
            let execution_mode =
                ExecutionModeOp::new(ctx, func_name, ExecutionMode::LocalSize, vec![x, y, z]);
            execution_mode.get_operation().insert_at_front(block, ctx);
        },
    );
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

#[cfg(feature = "pliron-dump")]
pub(crate) fn dump_spirv(repr: &SpirvKernel, name: &str) {
    use std::fs;

    if let Some(dir) = kernel_dir_name(name) {
        let kernel = &repr.assembled_module;
        let kernel = kernel
            .iter()
            .flat_map(|it| it.to_le_bytes())
            .collect::<Vec<_>>();
        let kernel_path = dir.join("module.spv");
        fs::write(kernel_path, kernel).unwrap();
    }
}

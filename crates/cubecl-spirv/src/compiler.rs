use crate::{
    CollectVerCapExtPass, ConvertArgsPass, PARAMS_NAME, SpirvKernel,
    builtin::{BUILTINS_NAME, LowerBuiltinsPass},
    lower::LowerOpsSpirvPass,
    ops::{
        branch::BranchToSpirvConversionPass, memory::lower_shared,
        to_spirv_dialect::ToSpirvDialectPass,
    },
    params_storage_class,
};
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    Compiler, WgpuCompilationOptions,
    ir::{
        ContextExt,
        attributes::{ATTR_READONLY, FuncInterface},
        ident,
        metadata::Info,
        rewrite::SimplifyOpsPass,
    },
    post_processing::{
        bitwise::PromoteBitwisePass, disaggregate::DisaggregatePass,
        saturating::LowerSaturatingArithmeticPass,
    },
    prelude::{KernelDefinition, Visibility},
};
use cubecl_ir::{
    attributes::EntrypointInterface,
    prelude::{SingleBlockRegionInterface, SymbolOpInterface},
    rewrite::visit_all_ops_of_type_mut,
    settings::Dim3,
};
use cubecl_opt::passes::{
    alloc_shared_memory::AllocateSharedMemoryBlockPass,
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, simple_cse::SimpleCSEPass,
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
    opts::{
        constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass,
        simplify_cfg::SimplifyCFGPass,
    },
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
    printable::Printable,
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

        let entry_func = value.body.state().entry_func;
        let module = value.body.state().module;

        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty::<Info>(value.info);
        ctx.set_aux_ty::<WgpuCompilationOptions>(*compilation_options);
        ctx.set_aux_ty::<KernelInfo>(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });

        let entry = entry_func.get_entry_block(&ctx);
        let bindings = (0..entry.deref(&ctx).get_num_arguments()).map(|i| {
            let readonly = entry_func.has_arg_attr(&ctx, i, &ATTR_READONLY);
            match readonly {
                true => Visibility::Read,
                false => Visibility::ReadWrite,
            }
        });
        let bindings: Vec<Visibility> = bindings.collect();

        let info_visibility = Visibility::Read;
        let immediate_size = match params_storage_class(&ctx, bindings.len()) {
            StorageClass::PushConstant => Some((bindings.len() + 1) * size_of::<u64>()),
            _ => None,
        };

        let (module, shared_size) = self.compile_kernel(&mut ctx, module);

        Ok(SpirvKernel {
            assembled_module: module.assemble(),
            module: Some(Arc::new(module)),
            bindings,
            shared_size,
            immediate_size,
            info_visibility,
        })
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
    pub fn compile_kernel(&mut self, ctx: &mut Context, module: ModuleOp) -> (Module, usize) {
        let module_op = module.get_operation();

        std::fs::write("target/initial.plir", format!("{}", module.disp(ctx))).unwrap();
        verify_operation(module.get_operation(), ctx).expect("Failed to verify before passes");

        let config = PMConfig {
            print_after_all: true,
            // ir_printing_dir: Some("target".into()),
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(AllocateSharedMemoryBlockPass);
        func_passes.add_pass(LowerSaturatingArithmeticPass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(LowerBuiltinsPass);

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(LowerOpsSpirvPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);

        passes.run(module_op, ctx, &mut analyses).unwrap();

        std::fs::write(
            "target/after_lower_shared.plir",
            format!("{}", module.disp(ctx)),
        )
        .unwrap();

        verify_operation(module_op, ctx).expect("Failed to verify after passes");

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(BranchToSpirvConversionPass::default());
        // func_passes.add_pass(Mem2RegPass);
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimplifyCFGPass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.run(module_op, ctx, &mut analyses).unwrap();

        std::fs::write(
            "target/after_lower_cfg.plir",
            format!("{}", module.disp(ctx)),
        )
        .unwrap();

        verify_operation(module_op, ctx).expect("Failed to verify after passes");

        let spirv_module = insert_spirv_module(ctx, module);
        let spirv_module_op = spirv_module.get_operation();

        let mut passes = OpPass::<SpirvModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(DCEPass);
        func_passes.add_pass(ToSpirvDialectPass::default());

        passes.add_pass(ConvertArgsPass);
        passes.add_pass(NestedOpsPass::new(func_passes));

        passes.run(spirv_module_op, ctx, &mut analyses).unwrap();

        let (shared_size, shared_args) = lower_shared(ctx, spirv_module);
        declare_entry_point(ctx, spirv_module, shared_args);

        // Make sure this is the last pass so it catches all ops
        OpPass::<SpirvModuleOp, CollectVerCapExtPass>::default()
            .run(spirv_module_op, ctx, &mut analyses)
            .unwrap();

        std::fs::write(
            "target/after_convert_args.plir",
            format!("{}", module.disp(ctx)),
        )
        .unwrap();

        // verify_operation(module_op, ctx).expect("Failed to verify after passes");

        let mut builder = PlironBuilder::default();
        spirv_module
            .to_spirv(ctx, &mut builder)
            .expect("Failed to convert");
        let module = builder.module();

        (module, shared_size)
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

// pub(crate) fn convert_math_mode(math_mode: EnumSet<FastMath>) -> FPFastMathMode {
//     let mut flags = FPFastMathMode::NONE;

//     for mode in math_mode.iter() {
//         match mode {
//             FastMath::NotNaN => flags |= FPFastMathMode::NOT_NAN,
//             FastMath::NotInf => flags |= FPFastMathMode::NOT_INF,
//             FastMath::UnsignedZero => flags |= FPFastMathMode::NSZ,
//             FastMath::AllowReciprocal => flags |= FPFastMathMode::ALLOW_RECIP,
//             FastMath::AllowContraction => flags |= FPFastMathMode::ALLOW_CONTRACT,
//             FastMath::AllowReassociation => flags |= FPFastMathMode::ALLOW_REASSOC,
//             FastMath::AllowTransform => {
//                 flags |= FPFastMathMode::ALLOW_CONTRACT
//                     | FPFastMathMode::ALLOW_REASSOC
//                     | FPFastMathMode::ALLOW_TRANSFORM
//             }
//             _ => {}
//         }
//     }

//     flags
// }

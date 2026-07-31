use crate::{
    cuda::{mma::CudaCmmaCompiler, packed_ops::PackOpsPass},
    shared::{
        builtin::{LowerBuiltins, LowerBuiltinsPass},
        convert::PromoteUnsupportedTypesPass,
        lowering::LowerOpsCppPass,
        metadata::LowerInfoPass,
        signature::{
            CollectIncludesPass, DeclareInfoTypeOp, DeclareVectorTypesPass, buffers,
            shared_memory_size,
        },
        unroll::CppUnrollPass,
    },
    target::{CppTarget, Shared, Target},
};

use super::ComputeKernel;
use core::marker::PhantomData;
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    ir::{
        AddressType, ContextExt, DeviceProperties, ElemType, FloatKind, IntKind, Type, UIntKind,
        features::{AtomicUsage, TypeUsage},
        metadata::Info,
        rewrite::SimplifyOpsPass,
        settings::Dim3,
    },
    post_processing::{
        bitwise::PromoteBitwisePass,
        checked_io::{CheckedIo, CheckedIoPass},
        disaggregate::DisaggregatePass,
        expression_merge::RemoveTrivialOpsPass,
        saturating::LowerSaturatingArithmeticPass,
    },
    prelude::KernelDefinition,
};
use cubecl_opt::passes::{
    alloc_shared_memory::AllocateSharedMemoryBlockPass,
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, simple_cse::SimpleCSEPass,
};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    context::Context,
    irbuild::match_rewrite::MatchRewrite,
    op::Op,
    operation::verify_operation,
    opts::{constants::sccp::SCCPPass, dce::DCEPass, mem2reg::Mem2RegPass},
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
};
use std::fmt::Debug;

pub(crate) fn closure_inference_hack<T, R>(
    val: &T,
    ctx: &Context,
    func: impl FnOnce(&T, &Context) -> R,
) -> R {
    func(val, ctx)
}

macro_rules! scoped_block {
    ($($lines: expr)*) => {{
        let mut out = String::from("[&]{\n");
        $(
            out.push_str(&$lines);
            out.push_str("\n");
        )*
        out.push_str("}()");
        out
    }};
}
pub(crate) use scoped_block;

#[derive(Clone, Copy, Debug)]
pub struct CompilationOptions {
    pub warp_size: usize,
    pub supports_features: CppSupportedFeatures,
}

pub struct CompilationState {
    pub cube_dim: Dim3,
    pub cluster_dim: Dim3,
    pub info: Info,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CppSupportedFeatures {
    pub grid_constants: bool,
    pub clusters: bool,
    pub fast_math: bool,
    pub fast_tanh: bool,
    pub elect_sync: bool,
}

impl Default for CompilationOptions {
    fn default() -> Self {
        Self {
            warp_size: 32,
            supports_features: Default::default(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CppCompiler<T: CppTarget> {
    _target: PhantomData<T>,
}

impl<T: CppTarget> Compiler for CppCompiler<T>
where
    LowerBuiltins<T>: MatchRewrite,
{
    type Representation = ComputeKernel;
    type CompilationOptions = CompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = kernel.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile cpp kernel\nCaused by:\n  ".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        self.compile_ir(kernel, *compilation_options)
    }

    fn extension(&self) -> &'static str {
        "cpp"
    }
}

impl<T: CppTarget> CppCompiler<T>
where
    LowerBuiltins<T>: MatchRewrite,
{
    fn compile_ir(
        self,
        kernel: KernelDefinition,
        compilation_options: CompilationOptions,
    ) -> Result<ComputeKernel, CompilationError> {
        let module = kernel.body.state().module;
        let module_op = module.get_operation();
        let entry_func = kernel.body.state().entry_func;
        let mut ctx = kernel.body.into_context().expect("Should be owned scope");

        let state = CompilationState {
            cube_dim: kernel.settings.cube_dim,
            cluster_dim: kernel.settings.cluster_dim.unwrap_or(Dim3::new_single()),
            info: kernel.info,
        };

        ctx.set_aux_ty(compilation_options);
        ctx.set_aux_ty(state);
        ctx.set_aux_ty(T::target());

        ctx.set_aux_ty(CudaCmmaCompiler::Cpp);

        verify_operation(module.get_operation(), &ctx)?;

        // This is an op so it can be inserted after the includes, which is important for scalars
        // that need includes. I wish C++ didn't have ordering dependent declarations...
        let decl_types = DeclareInfoTypeOp::new(&mut ctx);
        decl_types
            .get_operation()
            .insert_before(&ctx, entry_func.get_operation());

        let config = PMConfig {
            ir_printing_dir: kernel_dir_name(&kernel.settings.kernel_name),
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(LowerInfoPass);
        func_passes.add_pass(DisaggregatePass);
        func_passes.add_pass(CheckedIoPass::new(CheckedIo::new(
            kernel.settings.execution_mode,
            kernel.settings.kernel_name,
        )));
        func_passes.add_pass(AllocateSharedMemoryBlockPass);

        // Shared lowerings can create ops that need target-specific lowerings, but target-specific
        // lowerings should take priority. So we just run the target-specific lowerings twice.
        func_passes.add_pass(LowerOpsCppPass::<T>::default());
        func_passes.add_pass(LowerOpsCppPass::<Shared>::default());
        func_passes.add_pass(LowerOpsCppPass::<T>::default());

        if T::target() != Target::Metal {
            func_passes.add_pass(LowerSaturatingArithmeticPass::default());
        }

        func_passes.add_pass(RemoveTrivialOpsPass::default());

        if T::target() == Target::Cuda {
            func_passes.add_pass(PackOpsPass::default());
        }

        func_passes.add_pass(CppUnrollPass::default());
        func_passes.add_pass(LowerBuiltinsPass::<T>::default());

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

        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(PromoteUnsupportedTypesPass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);
        passes.add_pass(DeclareVectorTypesPass);
        passes.add_pass(CollectIncludesPass::<T>::default());

        passes.run(module_op, &mut ctx, &mut analyses).unwrap();

        #[cfg(feature = "metal")]
        if T::target() == Target::Metal {
            crate::metal::builtin::append_msl_builtins(&mut ctx, entry_func);
        }

        verify_operation(module.get_operation(), &ctx)?;

        let shared_memory_size = shared_memory_size(&ctx, module_op);
        let buffers = buffers(&ctx, entry_func);

        Ok(ComputeKernel {
            ctx,
            shared_memory_size,
            buffers,
        })
    }
}

pub fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let supported_types = [
        ElemType::Index,
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::Flex32),
        ElemType::Float(FloatKind::F64),
        ElemType::Bool,
    ];

    let supported_atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F32),
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in supported_atomic_types {
        // Restricted to 32-bit integers because not every min/max/bitwise/CAS overload
        // exists for 64-bit and float atomics across the C++ dialects (CUDA, HIP, Metal).
        let usage = match ty {
            ElemType::Int(IntKind::I32) | ElemType::UInt(UIntKind::U32) => AtomicUsage::all(),
            _ => AtomicUsage::Add | AtomicUsage::LoadStore | AtomicUsage::Exchange,
        };
        props.register_atomic_type_usage(Type::atomic(ty), usage);
    }
}

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

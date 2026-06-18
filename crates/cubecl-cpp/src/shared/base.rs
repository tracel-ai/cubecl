use crate::{
    shared::{
        OpToCPP,
        builtin::{LowerBuiltins, LowerBuiltinsPass},
        lowering::LowerOpsCppPass,
        signature::LowerInfoPass,
        ty::ConvertPtrPass,
    },
    target::{CppTarget, Shared},
};

use super::ComputeKernel;
use core::marker::PhantomData;
use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    ir::{
        AddressType, ContextExt, DeviceProperties, ElemType, FloatKind, IntKind, Type, UIntKind,
        features::{AtomicUsage, TypeUsage},
        metadata::Info,
        settings::Dim3,
    },
    post_processing::disaggregate::DisaggregatePass,
    prelude::KernelDefinition,
};
use cubecl_runtime::compiler::{CompilationError, Compiler};
use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    common_traits::Verify,
    context::Context,
    irbuild::match_rewrite::MatchRewrite,
    op::Op,
    opts::constants::sccp::SCCPPass,
    pass_manager::{AnalysisManager, OpPass, OpPassManager, Pass, PassGroup},
    printable::Printable,
    r#type::TypeHandle,
    value::Value,
};
use std::{collections::HashMap, fmt::Debug};

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
    pub ext_meta_positions: HashMap<usize, usize>,
    pub address_type: TypeHandle,
    pub info_st: Option<Value>,
    pub dynamic_meta: Option<Value>,
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

/// Cube indexes flags.
/// When true the corresponding index is declared and computed as needed in the kernel.
#[derive(Debug, Clone, Default)]
pub struct CubeIndexFlags {
    pub absolute_pos: bool,
    pub absolute_pos_tuple: bool,
    pub cube_count: bool,
    pub cube_count_tuple: bool,
    pub cube_dim: bool,
    pub cube_dim_tuple: bool,
    pub cube_pos: bool,
    pub cube_pos_tuple: bool,
    pub plane_dim: bool,
    pub plane_pos: bool,
    pub unit_pos: bool,
    pub unit_pos_tuple: bool,
    pub unit_pos_plane: bool,
    pub cluster_pos: bool,
}

/// Flags gathered during Cube IR translation for the kernel compilation.
#[derive(Debug, Clone, Default)]
pub struct Flags {
    pub elem_fp4: bool,
    pub elem_fp6: bool,
    pub elem_fp8: bool,
    pub elem_bf16: bool,
    pub elem_f16: bool,
    pub elem_tf32: bool,
    pub indexes: CubeIndexFlags,
    pub op_barrier: bool,
    pub thread_block: bool,
    pub inst_tma: bool,
    pub inst_tma_im2col: bool,
    pub inst_wmma: bool,
    pub inst_ptx_wrappers: bool,
    pub inst_async_copy: bool,
    pub use_grid_constants: bool,
    pub static_meta_length: usize,
    pub has_dynamic_meta: bool,
    pub has_info: bool,
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Debug, Default)]
pub struct CppCompiler<T: CppTarget> {
    compilation_options: CompilationOptions,
    info: Info,
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

        self.compilation_options = *compilation_options;

        let ir = self.clone().compile_ir(kernel);
        Ok(ir)
    }

    fn extension(&self) -> &'static str {
        "cpp"
    }
}

impl<T: CppTarget> CppCompiler<T>
where
    LowerBuiltins<T>: MatchRewrite,
{
    fn compile_ir(self, value: KernelDefinition) -> ComputeKernel {
        // let mut opt = Optimizer::shared_only(value.body.clone(), value.cube_dim);
        // let shared_allocs = opt.main.analysis::<SharedLiveness>(&opt.global_state);

        // *value.body.state_mut() = scope_state;

        // CheckedIoVisitor::new(self.strategy, self.kernel_name.clone()).apply(&value.body);
        // DisaggregateVisitor::apply(&value.body);

        // self.buffer_vis = post_processing::optimize_scope(&value.body).into();
        // self.buffer_vis
        //     .resize(value.num_global_buffers(), Visibility::Read);

        // let address_type = address_type.to_type(ctx);
        // let instructions = self.compile_scope(&value.body);

        // let tensor_maps = value.tensor_maps.clone();
        // let buffers = value.buffers.clone();
        // let scalars = value
        //     .scalars
        //     .into_iter()
        //     .map(|binding| (binding.ty.to_type(ctx), binding.count))
        //     .collect::<Vec<_>>();

        // let shared_memories = shared_allocs
        //     .allocations
        //     .values()
        //     .map(|alloc| SharedMemory {
        //         ptr: self.compile_value(ExpandValue::new(alloc.id, alloc.smem.root_ptr.ty)),
        //         value_ty: self.compile_type(alloc.smem.value_ty),
        //         align: alloc.smem.alignment,
        //         offset: alloc.offset,
        //     })
        //     .collect();

        let module = value.body.state().module;
        let mut ctx = value.body.into_context().expect("Should be owned scope");

        let state = CompilationState {
            cube_dim: value.settings.cube_dim,
            cluster_dim: value.settings.cluster_dim.unwrap_or(Dim3::new_single()),
            info: value.info,
            ext_meta_positions: Default::default(),
            address_type: value.settings.address_type.unsigned_type().to_type(&ctx),
            info_st: Default::default(),
            dynamic_meta: Default::default(),
        };

        ctx.set_aux_ty(self.compilation_options);
        ctx.set_aux_ty(state);
        ctx.set_aux_ty(T::target());

        std::fs::write("target/initial.plir", format!("{}", module.disp(&ctx))).unwrap();

        let mut analyses = AnalysisManager::default();
        let mut pass_manager = OpPassManager::<ModuleOp>::default();

        pass_manager.add_pass(OpPass::<LowerInfoPass, FuncOp>::default());
        pass_manager.add_pass(OpPass::<LowerBuiltinsPass<T>, FuncOp>::default());
        pass_manager.add_pass(OpPass::<DisaggregatePass, FuncOp>::default());
        pass_manager.add_pass(OpPass::<SCCPPass, FuncOp>::default());
        pass_manager.add_pass(OpPass::<ConvertPtrPass, FuncOp>::default());
        pass_manager.add_pass(OpPass::<LowerOpsCppPass<T>, FuncOp>::default());
        pass_manager.add_pass(OpPass::<LowerOpsCppPass<Shared>, FuncOp>::default());

        pass_manager
            .run(module.get_operation(), &mut ctx, &mut analyses)
            .unwrap();
        std::fs::write(
            "target/after_lower_shared.plir",
            format!("{}", module.disp(&ctx)),
        )
        .unwrap();

        module.verify(&ctx).unwrap();
        module.to_cpp(&ctx);

        ComputeKernel {
            ctx,
            module,
            shared_memories: vec![],
            shared_memory_size: 0,
            info_by_ptr: !self.compilation_options.supports_features.grid_constants,
            has_dynamic_meta: self.info.has_dynamic_meta,
        }
    }
}

pub fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let supported_types = [
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
        props.register_atomic_type_usage(
            Type::atomic(ty),
            AtomicUsage::Add | AtomicUsage::LoadStore,
        );
    }
}

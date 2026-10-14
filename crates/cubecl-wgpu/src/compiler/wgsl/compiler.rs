use super::shader::ComputeShader;
use crate::compiler::wgsl::{
    self, EnableFeaturesPass,
    builtin::{LowerBuiltinsPass, WorkgroupLayout},
    lower::LowerOpsWgslPass,
    metadata::declare_info,
    rewrite_args, shared_memory_size, types,
};
use cubecl_server::kernel::BufferIOAttr;

use cubecl_core::{
    WgpuCompilationOptions,
    codegen::KernelDump,
    post_processing::{
        checked_io::{CheckedIo, CheckedIoPass},
        minifloat::{
            Fp8Container, LowerMinifloatCast, LowerMinifloatCastPass, LowerMinifloatCompare,
            LowerMinifloatComparePass,
        },
        saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::{
    ContextExt,
    features::EnumSet,
    pliron::{
        builtin::ops::{FuncOp, ModuleOp},
        operation::verify_operation,
        opts::{dce::DCEPass, mem2reg::Mem2RegPass},
        printable::Printable,
    },
    prelude::{AnalysisManager, NestedOpsPass, Op, OpPass, Pass, Passes},
    rewrite::SimplifyOpsPass,
    settings::Dim3,
};
use cubecl_opt::passes::{
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass, inst_combine::InstCombinePass,
    sccp::SCCPPass, simple_cse::SimpleCSEPass, sroa::SROAPass,
};
use cubecl_server::compiler::CompilationError;
use cubecl_server::kernel;

const MAX_VECTOR_SIZE: usize = 4;

#[derive(Clone, Copy)]
pub struct KernelInfo {
    pub cube_dim: Dim3,
    /// How the cube is laid out as a workgroup.
    pub layout: WorkgroupLayout,
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

    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        Some(repr.io.clone())
    }

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

    fn lang_tag(&self) -> &'static str {
        "wgsl"
    }
}

impl WgslCompiler {
    fn compile_shader(
        &mut self,
        value: kernel::KernelDefinition,
        compilation_options: &WgpuCompilationOptions,
    ) -> Result<wgsl::ComputeShader, CompilationError> {
        #[cfg(target_family = "wasm")]
        crate::compiler::wgsl::ops::wasm_inventory_root();

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

        let dump = KernelDump::new(&value.settings.kernel_name);

        let module = value.body.state().module;
        let entry_func = value.body.state().entry_func;
        let module_op = module.get_operation();
        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty(value.info);
        let cube_dim = value.settings.cube_dim;
        let layout = WorkgroupLayout::new(&ctx, module_op, &cube_dim);
        ctx.set_aux_ty(KernelInfo { cube_dim, layout });
        ctx.set_aux_ty(*compilation_options);

        dump.write("initial.plir", || module_op.disp(&ctx).to_string());

        verify_operation(module_op, &ctx)?;
        types::check_fp8_lanes(&ctx, module_op)?;

        let mut analyses = AnalysisManager::default();
        analyses.set_config(dump.pass_config());

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(SROAPass);
        func_passes.add_pass(CheckedIoPass::new(CheckedIo::new(
            value.settings.execution_mode,
            value.settings.kernel_name.clone(),
        )));
        func_passes.add_pass(UnrollPass::new(MAX_VECTOR_SIZE));
        // After the unroll so that an fp8 vector is at most one word, see `types.rs`.
        func_passes.add_pass(LowerMinifloatCastPass::new(LowerMinifloatCast::new(
            EnumSet::empty(),
            Fp8Container::Words,
        )));
        func_passes.add_pass(LowerMinifloatComparePass::new(LowerMinifloatCompare::new(
            Fp8Container::Words,
        )));

        func_passes.add_pass(LowerOpsWgslPass::default());
        func_passes.add_pass(LowerSaturatingArithmeticPass::default());
        func_passes.add_pass(LowerBuiltinsPass);

        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(InstCombinePass::default());
        func_passes.add_pass(SimpleCSEPass::without_memory());
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SROAPass);

        // SCCP/DCE may unlock more mem2reg opportunities, and vice versa. So we do a sandwich.
        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SROAPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass::with_memory());
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);
        passes.add_pass(EnableFeaturesPass);

        passes.run(module_op, &mut ctx, &mut analyses)?;

        // Before `rewrite_args` erases the arguments and widens their
        // visibility for the shader.
        let io = cubecl_core::ir::attributes::buffer_io_by_position(&ctx, entry_func)
            .into_iter()
            .collect();
        let buffers = rewrite_args(&mut ctx, entry_func);
        declare_info(&mut ctx, entry_func, buffers.len());
        let shared_memory_size = shared_memory_size(&ctx, module_op);

        verify_operation(module.get_operation(), &ctx)?;

        Ok(ComputeShader {
            buffers,
            io,
            shared_memory_size,
            ctx,
        })
    }
}

#[cfg(test)]
mod tests {
    use cubecl_core::{self as cubecl, Compiler, prelude::*};
    use cubecl_ir::{metadata::Info, settings::Dim3};

    use cubecl_server::kernel::KernelDefinition;

    use super::WgslCompiler;

    /// Reads a unit's axes and nothing of its plane.
    #[cube]
    fn axes() {
        let _ = UNIT_POS_X + UNIT_POS_Y + UNIT_POS_Z;
    }

    /// Reads a unit's axes and its plane.
    #[cube]
    fn axes_and_plane() {
        let _ = UNIT_POS_X + UNIT_POS_Y + UNIT_POS_Z + PLANE_POS;
    }

    /// Reads a unit's axes and sums over its plane, without reading which plane or lane it is.
    #[cube]
    fn axes_and_plane_sum() {
        let _ = plane_sum(UNIT_POS_X + UNIT_POS_Y + UNIT_POS_Z);
    }

    /// The `@workgroup_size` line of the WGSL a kernel of `body` compiles to in cubes of
    /// `cube_dim`.
    fn workgroup_size(cube_dim: Dim3, body: fn(&Scope)) -> String {
        let settings = KernelSettings::new(cube_dim, ExecutionMode::Checked, AddressType::U32);
        let scope = Scope::root(settings.clone());
        body(&scope);
        let definition = KernelDefinition {
            body: scope,
            settings,
            info: Info::default(),
        };
        let shader = WgslCompiler
            .compile(definition, &Default::default())
            .expect("the kernel compiles");
        shader
            .to_string()
            .lines()
            .find(|line| line.starts_with("@compute"))
            .expect("an entry point")
            .to_string()
    }

    #[test]
    fn a_cube_spanning_y_without_planes_keeps_its_shape() {
        assert_eq!(
            workgroup_size(Dim3::new_3d(8, 4, 2), axes::expand),
            "@compute @workgroup_size(8, 4, 2)"
        );
    }

    #[test]
    fn a_cube_spanning_y_with_planes_launches_flat() {
        assert_eq!(
            workgroup_size(Dim3::new_3d(8, 4, 2), axes_and_plane::expand),
            "@compute @workgroup_size(64, 1, 1)"
        );
    }

    #[test]
    fn a_cube_spanning_y_with_plane_ops_but_no_plane_ids_keeps_its_shape() {
        assert_eq!(
            workgroup_size(Dim3::new_3d(8, 4, 2), axes_and_plane_sum::expand),
            "@compute @workgroup_size(8, 4, 2)"
        );
    }

    #[test]
    fn a_cube_along_x_with_planes_keeps_its_shape() {
        assert_eq!(
            workgroup_size(Dim3::new_1d(64), axes_and_plane::expand),
            "@compute @workgroup_size(64, 1, 1)"
        );
    }
}

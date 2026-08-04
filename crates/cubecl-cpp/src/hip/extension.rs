use cubecl_core::ir::{
    dialect::matrix::{CastOp, FillOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, StoreOp},
    interfaces::TypedExt,
    rewrite::visit_all_ops_of_type,
};
use pliron::{context::Context, context::Ptr, operation::Operation};

use super::mma::{
    HipCmmaCompiler, HipCmmaExt, WmmaCast, WmmaExecute, WmmaFill, WmmaLoad, WmmaStore,
};
use crate::shared::wmma_api_base::matrix_ty;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension {
    #[default]
    NoExtension,
    Wmma(WmmaExtension),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WmmaExtension {
    Fill(WmmaFill),
    Load(WmmaLoad),
    Execute(WmmaExecute),
    Store(WmmaStore),
    Cast(WmmaCast),
}

impl WmmaExtension {
    pub fn format_wmma(
        &self,
        f: &mut core::fmt::Formatter<'_>,
        ctx: &Context,
    ) -> core::fmt::Result {
        match self {
            WmmaExtension::Fill(fill) => fill.format_extension(f, ctx),
            WmmaExtension::Load(load) => load.format_extension(f, ctx),
            WmmaExtension::Execute(execute) => execute.format_extension(f, ctx),
            WmmaExtension::Store(store) => store.format_extension(f, ctx),
            WmmaExtension::Cast(cast) => cast.format_extension(f, ctx),
        }
    }
}

struct DisplayWmma<'a>(&'a WmmaExtension, &'a Context);

impl core::fmt::Display for DisplayWmma<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.format_wmma(f, self.1)
    }
}

/// The intrinsic wmma compiler builds fragments out of `ext_vector_type` typedefs and calls into
/// `__device__` helpers it generates itself
pub fn compile_wmma_extensions(ctx: &Context, module: Ptr<Operation>) -> String {
    let mut extensions = Vec::new();

    // Manual mma drives `__builtin_amdgcn_wmma_*` on raw register vectors rather than going through
    // the fragment API, so it needs the intrinsic helper whichever cmma compiler is selected.
    visit_all_ops_of_type::<MmaManualOp, _>(ctx, &mut extensions, module, |ctx, exts, op| {
        let execute = WmmaExecute::from_manual(
            op.shape(ctx).0,
            op.registers_a(ctx).scalar_ty(ctx),
            op.registers_c(ctx).scalar_ty(ctx),
        );
        push(exts, WmmaExtension::Execute(execute));
    });

    // rocWMMA lowers the fragment ops straight to library calls, so they need no helpers.
    if !matches!(ctx.hip_cmma(), HipCmmaCompiler::Intrinsics) {
        return finish(ctx, &extensions);
    }

    visit_all_ops_of_type::<FillOp, _>(ctx, &mut extensions, module, |ctx, exts, op| {
        let frag = matrix_ty(ctx, op.matrix(ctx));
        push(exts, WmmaExtension::Fill(WmmaFill::new(frag)));
    });
    visit_all_ops_of_type::<LoadOp, _>(ctx, &mut extensions, module, |ctx, exts, op| {
        let frag = matrix_ty(ctx, op.matrix(ctx));
        push(
            exts,
            WmmaExtension::Load(WmmaLoad::new(frag, op.layout(ctx).0)),
        );
    });
    visit_all_ops_of_type::<StoreOp, _>(ctx, &mut extensions, module, |ctx, exts, op| {
        let frag = matrix_ty(ctx, op.matrix(ctx));
        push(
            exts,
            WmmaExtension::Store(WmmaStore::new(frag, op.layout(ctx).0)),
        );
    });
    visit_all_ops_of_type::<MultiplyAccumulateOp, _>(
        ctx,
        &mut extensions,
        module,
        |ctx, exts, op| {
            let execute = WmmaExecute::new(
                matrix_ty(ctx, op.mat_a(ctx)),
                matrix_ty(ctx, op.mat_b(ctx)),
                matrix_ty(ctx, op.mat_c(ctx)),
                matrix_ty(ctx, op.mat_d(ctx)),
            );
            push(exts, WmmaExtension::Execute(execute));
        },
    );
    visit_all_ops_of_type::<CastOp, _>(ctx, &mut extensions, module, |ctx, exts, op| {
        let cast = WmmaCast::new(
            matrix_ty(ctx, op.input(ctx)),
            matrix_ty(ctx, op.output(ctx)),
        );
        push(exts, WmmaExtension::Cast(cast));
    });

    finish(ctx, &extensions)
}

fn finish(ctx: &Context, extensions: &[WmmaExtension]) -> String {
    if extensions.is_empty() {
        return String::new();
    }

    let mut out = HipCmmaCompiler::Intrinsics.type_definitions();
    for ext in extensions {
        out.push_str(&DisplayWmma(ext, ctx).to_string());
    }
    out
}

fn push(extensions: &mut Vec<WmmaExtension>, ext: WmmaExtension) {
    if !extensions.contains(&ext) {
        extensions.push(ext);
    }
}

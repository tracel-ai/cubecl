use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape},
    ir::{
        DeviceProperties,
        features::{MmaConfig, ScaledMmaConfig},
    },
};

pub type SupportedMmaCombinations = Vec<MmaConfig>;
pub type SupportedScaledMmaCombinations = Vec<ScaledMmaConfig>;

pub trait Architecture {
    fn warp_size(&self) -> u32;
    fn is_wmma_capable(&self) -> bool;
    fn is_mfma_capable(&self) -> bool;
    fn get_version(&self) -> u32 {
        0
    }
}

pub fn register_wmma_features(
    supported_combinations: SupportedMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.cmma.insert(config);
    }
}

pub fn register_mma_features(
    supported_combinations: SupportedMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.mma.insert(config);
    }
}

pub fn register_scaled_mma_features(
    supported_combinations: SupportedScaledMmaCombinations,
    properties: &mut DeviceProperties,
) {
    for config in supported_combinations {
        properties.features.matmul.scaled_mma.insert(config);
    }
}

pub mod wmma_api_base {
    use cubecl_core::{
        cmma::{MatrixIdent, MatrixLayout, MatrixType},
        ir::{
            dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
            interfaces::{TypeExt, TypedExt},
            types::{PointerType, scalar::TFloat32Type},
        },
    };
    use pliron::{
        context::Context,
        r#type::{TypeHandle, Typed},
        value::Value,
    };

    use crate::shared::{CppValue, ty::TypeExtCPP};

    use super::*;

    pub fn compile_matrix_declaration(ctx: &Context, val: Value, value_ty: TypeHandle) -> String {
        format!(
            "{} {id}_store; {} {id} = &{id}_store;",
            value_ty.to_cpp(ctx),
            val.get_type(ctx).to_cpp(ctx),
            id = val.name(ctx),
        )
    }

    pub fn compile_matrix(ctx: &Context, ty: &MatrixType, ns: &str) -> String {
        let elem = match ty.elem_ty.deref(ctx).is::<TFloat32Type>() {
            true => format!("{ns}::precision::tf32"),
            false => ty.elem_ty.to_cpp(ctx),
        };
        let ident = match ty.ident {
            MatrixIdent::A => format!("{ns}::matrix_a"),
            MatrixIdent::B => format!("{ns}::matrix_b"),
            MatrixIdent::Accumulator => format!("{ns}::accumulator"),
        };
        let MatrixShape { m, n, k } = ty.shape;
        let layout = match ty.layout {
            MatrixLayout::ColMajor => format!("{ns}::col_major"),
            MatrixLayout::RowMajor => format!("{ns}::row_major"),
            MatrixLayout::Undefined => {
                return format!("{ns}::fragment<{ident}, {m}, {n}, {k}, {elem}>");
            }
        };
        format!("{ns}::fragment<{ident}, {m}, {n}, {k}, {elem}, {layout}>")
    }

    pub fn fill(ctx: &Context, op: &FillOp, namespace: &str) -> String {
        let mat = op.matrix(ctx).name(ctx);
        let value = op.value(ctx).name(ctx);
        format!("{namespace}::fill_fragment(*{mat}, {value});")
    }

    pub fn load(ctx: &Context, op: &LoadOp, namespace: &str) -> String {
        let mat = op.matrix(ctx).name(ctx);
        let stride = op.stride(ctx).name(ctx);
        let ptr = as_scalar_ptr(ctx, op.source(ctx));
        let layout = match op.layout(ctx).0 {
            MatrixLayout::RowMajor => format!(", {namespace}::mem_row_major"),
            MatrixLayout::ColMajor => format!(", {namespace}::mem_col_major"),
            _ => String::new(),
        };
        format!("{namespace}::load_matrix_sync(*{mat}, {ptr}, {stride}{layout});")
    }

    pub fn store(ctx: &Context, op: &StoreOp, namespace: &str) -> String {
        let mat = op.matrix(ctx).name(ctx);
        let stride = op.stride(ctx).name(ctx);
        let destination = as_scalar_ptr(ctx, op.destination(ctx));
        let layout = op.layout(ctx).0;
        let layout = match layout {
            MatrixLayout::ColMajor => format!("{namespace}::mem_col_major"),
            MatrixLayout::RowMajor => format!("{namespace}::mem_row_major"),
            _ => unreachable!(),
        };

        format!("{namespace}::store_matrix_sync({destination}, *{mat}, {stride}, {layout});")
    }

    pub fn execute(ctx: &Context, op: &MultiplyAccumulateOp, namespace: &str) -> String {
        let mat_a = op.mat_a(ctx).name(ctx);
        let mat_b = op.mat_b(ctx).name(ctx);
        let mat_c = op.mat_c(ctx).name(ctx);
        let mat_d = op.mat_d(ctx).name(ctx);

        format!("{namespace}::mma_sync(*{mat_d}, *{mat_a}, *{mat_b}, *{mat_c});")
    }

    pub fn cast(ctx: &Context, op: &CastOp) -> String {
        let input = op.input(ctx).name(ctx);
        let output = op.output(ctx).name(ctx);
        let out_ty = op.output(ctx).unwrap_ptr(ctx).deref(ctx);
        let mat_ty = out_ty.downcast_ref::<MatrixType>().unwrap();
        let out_elem = mat_ty.elem_ty.to_cpp(ctx);
        format!(
            "for(int t=0; t<{input}->num_elements; t++) {{ {output}->x[t] = {out_elem}({input}->x[t]); }}"
        )
    }

    fn as_scalar_ptr(ctx: &Context, value: Value) -> String {
        let PointerType {
            inner,
            address_space,
        } = value.get_type(ctx).as_ptr(ctx);
        let new_ty = PointerType::get(ctx, inner.scalar_ty(ctx), address_space).to_handle();
        format!(
            "reinterpret_cast<{}>({})",
            new_ty.to_cpp(ctx),
            value.name(ctx)
        )
    }
}

pub fn frag_ident_str(frag: &MatrixIdent) -> &str {
    match frag {
        MatrixIdent::A => "a",
        MatrixIdent::B => "b",
        MatrixIdent::Accumulator => "c",
    }
}

pub fn frag_layout_str(frag: &MatrixLayout) -> &str {
    match frag {
        MatrixLayout::ColMajor => "col",
        MatrixLayout::RowMajor => "row",
        MatrixLayout::Undefined => "",
    }
}

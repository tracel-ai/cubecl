use cubecl_core::{
    self as cubecl,
    cmma::{MatrixIdent, MatrixType},
    ir::{
        ElemType, FloatKind,
        dialect::matrix::{ColIndexOp, MmaManualOp, RowIndexOp},
        features::MmaConfig,
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};
use itertools::Itertools;
use pliron::{
    context::Context,
    r#type::{Type, TypedHandle},
    value::Value,
};

use crate::{
    cuda::mma::manual::frag_elem,
    hip::{arch::AMDArchitecture, hip_op, mma::WmmaExecute},
    shared::{
        Architecture, CppValue, SupportedMmaCombinations, SupportedScaledMmaCombinations,
        lowering::LowerOp, ty::TypeExtCPP,
    },
    target::Hip,
};

#[op_interface_impl]
impl LowerOp<Hip> for RowIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ident = self.matrix_ty(scope.ctx()).deref(scope.ctx()).ident;
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        vec![row_index::expand(scope, lane_id.into(), i.into(), ident).value(scope)]
    }
}

#[op_interface_impl]
impl LowerOp<Hip> for ColIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ident = self.matrix_ty(scope.ctx()).deref(scope.ctx()).ident;
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        vec![col_index::expand(scope, lane_id.into(), i.into(), ident).value(scope)]
    }
}

#[cube]
fn row_index(lane_id: u32, i: u32, #[comptime] ident: MatrixIdent) -> u32 {
    match ident {
        MatrixIdent::A => lane_id % 16,
        MatrixIdent::B => i,
        // 2 * i, offset by 1 if lane_id >= 16
        MatrixIdent::Accumulator => i * 2 + (lane_id / 16),
    }
}

#[cube]
fn col_index(lane_id: u32, i: u32, #[comptime] ident: MatrixIdent) -> u32 {
    match ident {
        MatrixIdent::A => i,
        MatrixIdent::B => lane_id % 16,
        MatrixIdent::Accumulator => lane_id % 16,
    }
}

hip_op!(MmaManualOp, compile_manual_mma);

pub(super) fn compile_manual_mma(op: &MmaManualOp, ctx: &Context) -> String {
    let frag_a = op.registers_a(ctx);
    let frag_b = op.registers_b(ctx);
    let frag_c = op.registers_c(ctx);
    let frag_d = op.registers_d(ctx);
    let shape = op.shape(ctx).0;

    let elem_a = frag_elem(ctx, frag_a);
    let elem_c = frag_elem(ctx, frag_c);
    let elem_d = frag_elem(ctx, frag_d).to_cpp(ctx);

    let extension = WmmaExecute::from_manual(shape, elem_a, elem_c);

    let cd_elems = shape.num_elems(MatrixIdent::Accumulator) / 32;

    let frag_cd_step = 4usize.div_ceil(elem_c.size(ctx));

    // Need to reconstruct the fragments from an array of vectors to a single vector type.
    // `float8_t {reinterpret_cast<const float*>(arr->data)[0], ...}`
    let frag = |val: Value, len: usize| {
        let elem = frag_elem(ctx, val).to_cpp(ctx);
        let ptr = format!("reinterpret_cast<const {elem}*>({}->data)", val.name(ctx));
        (0..len).map(|i| format!("{ptr}[{i}]")).join(", ")
    };

    let frag_a = frag(frag_a, 16);
    let frag_b = frag(frag_b, 16);
    // C matrix needs to be padded for f16, because it only uses the low bytes. The simplest way is
    // to just replicate the same f16 in both halves of the register.
    let frag_c = {
        let elem = elem_c.to_cpp(ctx);
        let frag_c = frag_c.name(ctx);
        let ptr = format!("reinterpret_cast<const {elem}*>({frag_c}->data)");
        (0..cd_elems)
            .flat_map(|i| {
                let ptr = ptr.clone();
                (0..frag_cd_step).map(move |_| format!("{ptr}[{i}]"))
            })
            .join(", ")
    };

    // Should optimize out
    let name = extension.fn_name(ctx);

    let mut out = String::from("{{");
    out.push_str(&format!(
        "{{{} frag_d_tmp = {{}};",
        extension.frag_d.get_self_handle(ctx).to_cpp(ctx)
    ));

    out.push_str(&format!(
        "{name}({}{{{frag_a}}}, {}{{{frag_b}}}, {}{{{frag_c}}}, frag_d_tmp);",
        extension.frag_a.get_self_handle(ctx).to_cpp(ctx),
        extension.frag_b.get_self_handle(ctx).to_cpp(ctx),
        extension.frag_c.get_self_handle(ctx).to_cpp(ctx)
    ));

    let frag_d_ptr = format!("reinterpret_cast<{elem_d}*>({}->data)", frag_d.name(ctx));

    for i in 0..cd_elems {
        out.push_str(&format!(
            "{frag_d_ptr}[{i}] = frag_d_tmp[{i} * {frag_cd_step}];"
        ));
    }

    out.push_str("}}");

    out
}

pub fn supported_mma_combinations(arch: &AMDArchitecture) -> SupportedMmaCombinations {
    // Correctness is wrong.
    const ENABLED: bool = true;

    if !ENABLED {
        return Vec::new();
    }

    // Reference: https://gpuopen.com/learn/wmma_on_rdna3/
    // Feel free to add more if additional intrinsics are supported for execute
    let mut result: SupportedMmaCombinations = vec![];
    if arch.is_wmma_capable() {
        // Types fully supported.
        let types = vec![
            (
                ElemType::Float(FloatKind::F16),
                ElemType::Float(FloatKind::F32),
            ),
            (
                ElemType::Float(FloatKind::BF16),
                ElemType::Float(FloatKind::F32),
            ),
        ];
        let combinations = types.into_iter().map(|(ab_elem, cd_elem)| MmaConfig {
            a_type: ab_elem.into(),
            b_type: ab_elem.into(),
            cd_type: cd_elem.into(),
            m: 16,
            n: 16,
            k: 16,
        });
        result.extend(combinations);
    }
    result
}

pub fn supported_scaled_mma_combinations(
    _arch: &AMDArchitecture,
) -> SupportedScaledMmaCombinations {
    vec![]
}

pub fn contiguous_elements_rdna3(
    ctx: &Context,
    ident: MatrixIdent,
    matrix: TypedHandle<MatrixType>,
) -> usize {
    // Don't exceed swizzle atom and load width
    let max_vector_size = 16 / matrix.deref(ctx).elem_ty.size(ctx);
    match ident {
        MatrixIdent::A | MatrixIdent::B => 16.min(max_vector_size),
        MatrixIdent::Accumulator => 1,
    }
}

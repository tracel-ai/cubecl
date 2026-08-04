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
use pliron::{context::Context, r#type::TypedHandle, value::Value};

use crate::{
    hip::{
        arch::{AMDArchitecture, AmdWmma},
        hip_op,
        mma::{WmmaExecute, amd_wmma, compile_fragment_intrinsic},
    },
    shared::{
        CppValue, SupportedMmaCombinations, SupportedScaledMmaCombinations, lowering::LowerOp,
        ty::TypeExtCPP,
    },
    target::Hip,
};

#[op_interface_impl]
impl LowerOp<Hip> for RowIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let (ident, m, k) = (matrix.ident, matrix.shape.m as u32, matrix.shape.k as u32);
        let rdna4 = amd_wmma(scope.ctx()) == AmdWmma::Rdna4;
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        vec![row_index::expand(scope, lane_id.into(), i.into(), ident, m, k, rdna4).value(scope)]
    }
}

#[op_interface_impl]
impl LowerOp<Hip> for ColIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let (ident, k) = (matrix.ident, matrix.shape.k as u32);
        let rdna4 = amd_wmma(scope.ctx()) == AmdWmma::Rdna4;
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        vec![col_index::expand(scope, lane_id.into(), i.into(), ident, k, rdna4).value(scope)]
    }
}

/// RDNA4 splits a dimension between lanes 0-15 and 16-31, giving each lane a contiguous half of
/// it. Every RDNA4 fragment is built this way: A and B split `k`, the accumulator splits `m`.
#[cube]
fn split_half(lane_id: u32, i: u32, #[comptime] dim: u32) -> u32 {
    (lane_id / 16) * comptime![dim / 2] + i
}

#[cube]
fn row_index(
    lane_id: u32,
    i: u32,
    #[comptime] ident: MatrixIdent,
    #[comptime] m: u32,
    #[comptime] k: u32,
    #[comptime] rdna4: bool,
) -> u32 {
    match ident {
        MatrixIdent::A => lane_id % 16,
        MatrixIdent::B => {
            if comptime![rdna4] {
                split_half(lane_id, i, k)
            } else {
                // RDNA3 hands every lane the whole `k` range, duplicated across the lane halves.
                i
            }
        }
        MatrixIdent::Accumulator => {
            if comptime![rdna4] {
                split_half(lane_id, i, m)
            } else {
                // 2 * i, offset by 1 if lane_id >= 16
                i * 2 + (lane_id / 16)
            }
        }
    }
}

#[cube]
fn col_index(
    lane_id: u32,
    i: u32,
    #[comptime] ident: MatrixIdent,
    #[comptime] k: u32,
    #[comptime] rdna4: bool,
) -> u32 {
    match ident {
        MatrixIdent::A => {
            if comptime![rdna4] {
                split_half(lane_id, i, k)
            } else {
                i
            }
        }
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

    // `registers_a/b/c` are array values, only `registers_d` is a pointer (see `MmaManualOp`).
    let elem_a = frag_a.scalar_ty(ctx);
    let elem_c = frag_c.scalar_ty(ctx);
    let elem_d = frag_d.scalar_ty(ctx).to_cpp(ctx);

    let extension = WmmaExecute::from_manual(shape, elem_a, elem_c);

    let cd_elems = shape.num_elems(MatrixIdent::Accumulator) / 32;
    let ab_elems = amd_wmma(ctx).frag_ab_elems(shape.k);

    // RDNA3 spreads a 16 bit accumulator over 32 bit lanes, using only the low half of each, so
    // its elements sit at every other index. RDNA4 packs them densely.
    let frag_cd_step = match amd_wmma(ctx) {
        AmdWmma::Rdna3 => 4usize.div_ceil(elem_c.size(ctx)),
        AmdWmma::Rdna4 => 1,
    };

    // Need to reconstruct the fragments from an array of vectors to a single vector type.
    // `float8_t {reinterpret_cast<const float*>(arr->data)[0], ...}`
    let frag = |val: Value, len: usize| {
        let elem = val.scalar_ty(ctx).to_cpp(ctx);
        let ptr = format!("reinterpret_cast<const {elem}*>({}.data)", val.name(ctx));
        (0..len).map(|i| format!("{ptr}[{i}]")).join(", ")
    };

    let frag_a = frag(frag_a, ab_elems);
    let frag_b = frag(frag_b, ab_elems);
    // C matrix needs to be padded for f16, because it only uses the low bytes. The simplest way is
    // to just replicate the same f16 in both halves of the register.
    let frag_c = {
        let elem = elem_c.to_cpp(ctx);
        let frag_c = frag_c.name(ctx);
        let ptr = format!("reinterpret_cast<const {elem}*>({frag_c}.data)");
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
        "{} frag_d_tmp = {{}};",
        compile_fragment_intrinsic(ctx, &extension.frag_d)
    ));

    out.push_str(&format!(
        "{name}({}{{{frag_a}}}, {}{{{frag_b}}}, {}{{{frag_c}}}, frag_d_tmp);",
        compile_fragment_intrinsic(ctx, &extension.frag_a),
        compile_fragment_intrinsic(ctx, &extension.frag_b),
        compile_fragment_intrinsic(ctx, &extension.frag_c)
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
    if arch.wmma_generation().is_some() {
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
            a_type: ab_elem,
            b_type: ab_elem,
            cd_type: cd_elem,
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
    contiguous_elements(AmdWmma::Rdna3, ctx, ident, matrix)
}

pub fn contiguous_elements_rdna4(
    ctx: &Context,
    ident: MatrixIdent,
    matrix: TypedHandle<MatrixType>,
) -> usize {
    contiguous_elements(AmdWmma::Rdna4, ctx, ident, matrix)
}

fn contiguous_elements(
    wmma: AmdWmma,
    ctx: &Context,
    ident: MatrixIdent,
    matrix: TypedHandle<MatrixType>,
) -> usize {
    let matrix = matrix.deref(ctx);
    // Don't exceed swizzle atom and load width
    let max_vector_size = 16 / matrix.elem_ty.size(ctx);
    match ident {
        // Consecutive elements of a lane's fragment are consecutive `k`, so a lane can load as
        // many in one go as it holds.
        MatrixIdent::A | MatrixIdent::B => wmma.frag_ab_elems(matrix.shape.k).min(max_vector_size),
        MatrixIdent::Accumulator => 1,
    }
}

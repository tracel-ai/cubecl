use cubecl_core::{
    cmma::MatrixIdent,
    ir::{
        ElemType, FloatKind, dialect::matrix::MmaManualOp, features::MmaConfig,
        interfaces::TypedExt, prelude::*,
    },
};
use itertools::Itertools;
use pliron::{context::Context, value::Value};

use crate::{
    hip::{
        arch::{AMDArchitecture, AmdWmma},
        hip_op,
        mma::{WmmaExecute, amd_wmma, compile_fragment_intrinsic},
    },
    shared::{CppValue, SupportedMmaCombinations, SupportedScaledMmaCombinations, ty::TypeExtCPP},
};

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
    let ab_elems = amd_wmma(ctx).frag_ab_elems_at_depth(shape.k);

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

    // On RDNA4, A and B lay out their registers alike, each lane holding one row of A or one
    // column of B. Handing the instruction B as its A and A as its B computes the transposed
    // product, which leaves each lane one row of the accumulator: the same layout as A, so a
    // product can feed the next one as its A without leaving registers.
    let (frag_a, frag_b) = match amd_wmma(ctx) {
        AmdWmma::Rdna3 => (frag_a, frag_b),
        AmdWmma::Rdna4 => (frag_b, frag_a),
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

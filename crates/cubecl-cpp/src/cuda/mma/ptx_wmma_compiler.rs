use super::WMMA_MINIMUM_VERSION;
use crate::{
    cuda::{arch::CudaArchitecture, ptx::mma_ty},
    shared::{Architecture, CompilationOptions, CppValue, SupportedMmaCombinations},
};
use core::cell::Ref;
use cubecl_core::{
    cmma::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
    ir::{
        self as gpu, ContextExt,
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        features::MmaConfig,
        interfaces::TypedExt,
        types::scalar::{BFloat16Type, Float16Type, Float32Type, Float64Type, TFloat32Type},
    },
};
use pliron::{
    context::Context,
    r#type::{TypeHandle, Typed},
    value::Value,
};

fn matrix_ty(ctx: &Context, ty: impl Typed) -> Ref<'_, MatrixType> {
    let ty = ty.get_type(ctx).deref(ctx);
    Ref::map(ty, |ty| {
        ty.downcast_ref::<MatrixType>().expect("Should be matrix")
    })
}

pub(super) fn compile_matrix_declaration_ptx(
    ctx: &Context,
    val: Value,
    value_ty: TypeHandle,
) -> String {
    let matrix = matrix_ty(ctx, value_ty);
    let val = val.name(ctx);
    let reg_count = get_fragment_register_total_count(ctx, &matrix);
    let ty = if matrix.elem_ty.deref(ctx).is::<Float32Type>() {
        "float"
    } else if matrix.elem_ty.deref(ctx).is::<Float64Type>() {
        "double"
    } else {
        "uint32_t"
    };
    format!("{ty} {val}[{reg_count}];")
}

pub(super) fn compile_matrix_ptx(ctx: &Context, ty: &MatrixType) -> String {
    let ty = if ty.elem_ty.deref(ctx).is::<Float32Type>() {
        "float"
    } else if ty.elem_ty.deref(ctx).is::<Float64Type>() {
        "double"
    } else {
        "uint32_t"
    };
    format!("{ty}*")
}

pub(super) fn fill_ptx(ctx: &Context, op: &FillOp) -> String {
    let matrix = op.matrix(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let mat_ty = matrix_ty(ctx, op.matrix(ctx));
    let reg_count = get_fragment_register_total_count(ctx, &mat_ty);
    format!(
        "// fill
for (uint i = 0; i < uint({reg_count}); ++i) {{
  {matrix}[i] = {value};
}}
 "
    )
}

pub(super) fn load_ptx(ctx: &Context, op: &LoadOp) -> String {
    let ptr = op.source(ctx).name(ctx);
    let matrix = op.matrix(ctx);
    let stride = op.stride(ctx);
    let mat_ty = matrix_ty(ctx, matrix);
    let elem_ty = mat_ty.elem_ty.deref(ctx);
    let MatrixShape { m, n, k } = mat_ty.shape;
    // Important note: the current frontend has been designed around
    // CUDA wmma which is not optimal in the case of PTX wmma and mma
    // We choose here to use the layout defined in the fragment first,
    // if it is unknown and we look into the layout passed to the instruction.
    let layout = get_qualifier_from_layout(&op.layout(ctx).0);

    // instruction qualifiers
    let ty = mma_ty(ctx, &*elem_ty);
    let ident = match mat_ty.ident {
        MatrixIdent::A => "a",
        MatrixIdent::B => "b",
        MatrixIdent::Accumulator => "c",
    };
    let opcode = format!("wmma.load.{ident}.sync.aligned.{layout}.m{m}n{n}k{k}.{ty}");

    // constraints
    let mut reg_count = 0;
    let (regs_decl, out_constraints) =
        get_value_regs_decl_constraints(ctx, matrix, true, &mut reg_count);
    let buffer_reg = format_reg_and_inc(&mut reg_count);
    let (stride_reg, stride_constraint) =
        get_value_regs_decl_constraints(ctx, stride, false, &mut reg_count);
    format!(
        r#"// load
{{
asm volatile(
    "{opcode} "
    "{{{regs_decl}}}, [{buffer_reg}], {stride_reg};\n"
    : {out_constraints}
    : "l"({ptr}){stride_constraint}
);
}}
"#
    )
}

pub(super) fn store_ptx(ctx: &Context, op: &StoreOp) -> String {
    let matrix = op.matrix(ctx);
    let destination = op.destination(ctx).name(ctx);
    let stride = op.stride(ctx);
    let mat_ty = matrix_ty(ctx, matrix);
    let elem_ty = mat_ty.elem_ty.deref(ctx);
    let MatrixShape { m, n, k } = mat_ty.shape;

    // instruction qualifiers
    let layout = match op.layout(ctx).0 {
        MatrixLayout::ColMajor => "col",
        MatrixLayout::RowMajor => "row",
        _ => unreachable!(),
    };
    let ty = mma_ty(ctx, &*elem_ty);
    let opcode = format!("wmma.store.d.sync.aligned.{layout}.m{m}n{n}k{k}.{ty}");
    // constraints
    let mut reg_count = 0;
    let buffer_reg = format_reg_and_inc(&mut reg_count);
    // offset and stride can be passed as local const or as const scalar
    // we need to handle both cases correctly in the asm.
    let (stride_reg, stride_constraint) =
        get_value_regs_decl_constraints(ctx, stride, false, &mut reg_count);
    // we start at 2 because of the buffer address calculation
    let (regs_decl, in_constraints) =
        get_value_regs_decl_constraints(ctx, matrix, false, &mut reg_count);
    format!(
        r#"// store
asm volatile(
    "{opcode} "
    "[{buffer_reg}], {{{regs_decl}}}, {stride_reg};\n"
    :
    : "l"({destination}),
      {in_constraints}{stride_constraint}
);
"#
    )
}

pub(super) fn execute_ptx(ctx: &Context, op: &MultiplyAccumulateOp) -> String {
    let mat_a = op.mat_a(ctx);
    let mat_b = op.mat_b(ctx);
    let mat_c = op.mat_c(ctx);
    let mat_d = op.mat_d(ctx);
    let mat_a_ty = matrix_ty(ctx, mat_a);
    let elem_a_ty = mat_a_ty.elem_ty.deref(ctx);
    let MatrixShape { m, n, k } = mat_a_ty.shape;

    let layout_a = get_fragment_layout_qualifier(ctx, mat_a);
    let layout_b = get_fragment_layout_qualifier(ctx, mat_b);

    let type_a = mma_ty(ctx, &*mat_a.get_type(ctx).deref(ctx));
    let type_b = mma_ty(ctx, &*mat_b.get_type(ctx).deref(ctx));
    let type_c = mma_ty(ctx, &*mat_c.get_type(ctx).deref(ctx));
    let type_d = mma_ty(ctx, &*mat_d.get_type(ctx).deref(ctx));

    let types = if elem_a_ty.is::<Float16Type>() {
        format!("{type_d}.{type_c}")
    } else {
        format!("{type_d}.{type_a}.{type_b}.{type_c}")
    };
    let opcode = format!("wmma.mma.sync.aligned.m{m}n{n}k{k}.{layout_a}.{layout_b}.{types}");

    let mut reg_count = 0;
    // order matters, declare the registers in the same order as the intrinsic
    let (regs_decl_d, out_constraints_d) =
        get_value_regs_decl_constraints(ctx, mat_d, true, &mut reg_count);
    let (regs_decl_a, in_constraints_a) =
        get_value_regs_decl_constraints(ctx, mat_a, false, &mut reg_count);
    let (regs_decl_b, in_constraints_b) =
        get_value_regs_decl_constraints(ctx, mat_b, false, &mut reg_count);
    let (regs_decl_c, in_constraints_c) =
        get_value_regs_decl_constraints(ctx, mat_c, false, &mut reg_count);
    format!(
        r#"// execute
asm volatile(
    "{opcode} "
    "{{{regs_decl_d}}}, "
    "{{{regs_decl_a}}}, "
    "{{{regs_decl_b}}}, "
    "{{{regs_decl_c}}};\n"
    : {out_constraints_d}
    : {in_constraints_a}, {in_constraints_b}, {in_constraints_c}
);
"#
    )
}

pub(super) fn cast_ptx(ctx: &Context, op: &CastOp) -> String {
    let input = op.input(ctx);
    let output = op.output(ctx);
    let mat_ty = matrix_ty(ctx, input);

    let reg_count = get_fragment_register_total_count(ctx, &mat_ty);
    let out_ty = matrix_ty(ctx, output).elem_ty.deref(ctx);

    let input = input.name(ctx);
    let output = output.name(ctx);

    if out_ty.is::<Float16Type>() {
        format!(
            "// cast
for (int i = 0; i < {reg_count}; ++i) {{
    __half h_lo = __float2half_rn({input}[2*i + 0]);
    __half h_hi = __float2half_rn({input}[2*i + 1]);
    __half2 h2 = __halves2half2(h_lo, h_hi);
    {output}[i] = *reinterpret_cast<unsigned int*>(&h2);
}}
"
        )
    } else if out_ty.is::<BFloat16Type>() {
        format!(
            "// cast
for (int i = 0; i < {reg_count}; ++i) {{
    __nv_bfloat16 b_lo = __float2bfloat16({input}[2*i + 0]);
    __nv_bfloat16 b_hi = __float2bfloat16({input}[2*i + 1]);
    __nv_bfloat162 bf2 = __halves2bfloat162(b_lo, b_hi);
    {output}[i] = *reinterpret_cast<unsigned int*>(&bf2);
}}
"
        )
    } else {
        unreachable!()
    }
}

fn get_fragment_register_total_count(ctx: &Context, frag: &MatrixType) -> usize {
    let MatrixType {
        ident,
        shape,
        elem_ty,
        ..
    } = *frag;
    let elements = shape.num_elems(ident);

    let bits_per_elem = elem_ty.unpacked_size_bits(ctx);
    let warp_size = ctx.aux_ty::<CompilationOptions>().warp_size;
    let lanes_per_reg = warp_size / bits_per_elem;
    // choose threads-per-frag:
    // - accumulators always use 32 lanes
    // - A/B use 16 lanes _except_ TF32 (k=8) which also uses 32 lanes
    let threads_per_frag = match ident {
        MatrixIdent::Accumulator => 32,
        MatrixIdent::A | MatrixIdent::B => {
            if elem_ty.deref(ctx).is::<TFloat32Type>() {
                32
            } else {
                16
            }
        }
    };

    elements / (lanes_per_reg * threads_per_frag)
}

fn get_fragment_layout_qualifier(ctx: &Context, val: Value) -> String {
    let frag = matrix_ty(ctx, val);
    get_qualifier_from_layout(&frag.layout)
}

fn get_qualifier_from_layout(layout: &MatrixLayout) -> String {
    match layout {
        MatrixLayout::ColMajor => "col",
        MatrixLayout::RowMajor => "row",
        _ => unreachable!(),
    }
    .to_string()
}

fn get_value_regs_decl_constraints(
    ctx: &Context,
    val: Value,
    output: bool,
    reg_count: &mut u8,
) -> (String, String) {
    let ty = val.get_type(ctx).deref(ctx);
    let val = val.name(ctx);

    if let Some(mat_ty) = ty.downcast_ref::<MatrixType>() {
        let reg_total_count = get_fragment_register_total_count(ctx, mat_ty);
        let reg_decl = (0..reg_total_count)
            .map(|_| format_reg_and_inc(reg_count))
            .collect::<Vec<_>>()
            .join(",");
        let frag_elem = mat_ty.elem_ty;
        let modifier = format!(
            "{}{}",
            if output { "=" } else { "" },
            if frag_elem.is_float32(ctx) {
                "f"
            } else if frag_elem.is_float64(ctx) {
                "d"
            } else {
                "r"
            }
        );
        let constraints = (0..reg_total_count)
            .map(|i| format!("\"{modifier}\"({val}[{i}])"))
            .collect::<Vec<_>>()
            .join(", ");
        (reg_decl, constraints)
    } else {
        (format_reg_and_inc(reg_count), format!(r#", "r"({val})"#))
    }
}

fn format_reg_and_inc(count: &mut u8) -> String {
    let res = format!("%{count}");
    *count += 1;
    res
}

pub(super) fn supported_cmma_combinations_ptx(arch: &CudaArchitecture) -> SupportedMmaCombinations {
    let mut result: SupportedMmaCombinations = vec![];
    if arch.get_version() >= WMMA_MINIMUM_VERSION {
        // Types fully supported.
        let types = vec![
            (
                gpu::ElemType::Float(gpu::FloatKind::F16), // m
                gpu::ElemType::Float(gpu::FloatKind::F16), // n
                gpu::ElemType::Float(gpu::FloatKind::F16), // k
            ),
            (
                gpu::ElemType::Float(gpu::FloatKind::F16),
                gpu::ElemType::Float(gpu::FloatKind::F16),
                gpu::ElemType::Float(gpu::FloatKind::F32),
            ),
            (
                gpu::ElemType::Float(gpu::FloatKind::BF16),
                gpu::ElemType::Float(gpu::FloatKind::BF16),
                gpu::ElemType::Float(gpu::FloatKind::F32),
            ),
        ];
        let combinations: SupportedMmaCombinations = types
            .into_iter()
            .map(|(a, b, cd)| MmaConfig {
                a_type: a,
                b_type: b,
                cd_type: cd,
                m: 16,
                n: 16,
                k: 16,
            })
            .collect();
        result.extend(combinations);
        if arch.get_version() >= 72 {
            result.extend([
                MmaConfig {
                    a_type: gpu::ElemType::UInt(gpu::UIntKind::U8),
                    b_type: gpu::ElemType::UInt(gpu::UIntKind::U8),
                    cd_type: gpu::ElemType::Int(gpu::IntKind::I32),
                    m: 16,
                    n: 16,
                    k: 16,
                },
                MmaConfig {
                    a_type: gpu::ElemType::Int(gpu::IntKind::I8),
                    b_type: gpu::ElemType::Int(gpu::IntKind::I8),
                    cd_type: gpu::ElemType::Int(gpu::IntKind::I32),
                    m: 16,
                    n: 16,
                    k: 16,
                },
            ]);
        }
        if arch.get_version() >= 80 {
            result.push(MmaConfig {
                a_type: gpu::ElemType::Float(gpu::FloatKind::TF32),
                b_type: gpu::ElemType::Float(gpu::FloatKind::TF32),
                cd_type: gpu::ElemType::Float(gpu::FloatKind::F32),
                m: 16,
                n: 16,
                k: 8,
            });
        }
    }
    result
}

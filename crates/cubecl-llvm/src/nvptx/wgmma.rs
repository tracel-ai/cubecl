//! NVPTX warpgroup MMA, Hopper's asynchronous tensor core instructions.
//!
//! LLVM has intrinsics for the fence and for committing and waiting on groups, but none for
//! `wgmma.mma_async` itself, so the MMA is inline PTX, as MLIR's NVVM dialect emits it.

use super::matrix::{
    RegisterForm, as_shared, call_void, registers_into, registers_of, registers_result_ty,
    store_fragment, unpack_registers, vector_bits, vector_lanes,
};
use crate::{
    prelude::*,
    shared::matrix::{registers_array_ty, registers_as_vector, registers_value, vector_into_array},
};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    dialect::matrix::{
        WgmmaCommitGroupOp, WgmmaDescriptorOp, WgmmaFenceOp, WgmmaFenceOperandOp, WgmmaOp,
        WgmmaWaitGroupOp,
    },
    features::WgmmaConfig,
    types::{MatrixLayout, MatrixShape},
};
use pliron::r#type::type_cast;
use pliron_llvm::llvm_attrs::{LlvmAttrValue, LlvmAttributesAttr};
use std::sync::LazyLock;

const FENCE: &str = "llvm.nvvm.wgmma.fence.sync.aligned";
const COMMIT_GROUP: &str = "llvm.nvvm.wgmma.commit_group.sync.aligned";
const WAIT_GROUP: &str = "llvm.nvvm.wgmma.wait_group.sync.aligned";

/// Units of the four planes that issue one warpgroup MMA together.
const WARPGROUP_UNITS: u32 = 128;
/// Every warpgroup MMA computes 64 rows.
const M: u32 = 64;
const N_MAX: u32 = 256;
/// `A` held in registers is always four 32-bit registers per unit.
const A_REGISTERS: usize = 4;

#[derive(Debug, Error)]
#[error(
    "no warpgroup MMA multiplies {a:?} by {b:?} into {cd:?} with shape {shape:?}; supported \
     configurations: {supported:?}"
)]
pub struct WgmmaUnsupported {
    a: ElemType,
    b: ElemType,
    cd: ElemType,
    shape: MatrixShape,
    supported: Vec<WgmmaConfig>,
}

#[derive(Debug, Error)]
#[error(
    "a warpgroup MMA reads only K-major {0:?} tiles of {1:?}: the transposed layouts are for \
     16-bit elements"
)]
pub struct WgmmaLayoutUnsupported(&'static str, ElemType);

#[derive(Debug, Error)]
#[error(
    "the warpgroup MMA {what} holds {found} {unit} per unit, and the instruction takes {expected}"
)]
pub struct WgmmaRegisterCount {
    what: &'static str,
    unit: &'static str,
    found: usize,
    expected: usize,
}

#[derive(Debug, Error)]
#[error("a warpgroup MMA matrix descriptor is a u64, not {0}")]
pub struct WgmmaDescriptorType(String);

/// The warpgroup MMA forms Hopper (`sm_90a`) has.
pub fn wgmma_configs() -> &'static [WgmmaConfig] {
    static CONFIGS: LazyLock<Vec<WgmmaConfig>> = LazyLock::new(hopper_configs);
    &CONFIGS
}

fn hopper_configs() -> Vec<WgmmaConfig> {
    let f16 = ElemType::Float(FloatKind::F16);
    let bf16 = ElemType::Float(FloatKind::BF16);
    let tf32 = ElemType::Float(FloatKind::TF32);
    let f32 = ElemType::Float(FloatKind::F32);
    let e4m3 = ElemType::Float(FloatKind::E4M3);
    let e5m2 = ElemType::Float(FloatKind::E5M2);
    let s8 = ElemType::Int(IntKind::I8);
    let u8 = ElemType::UInt(UIntKind::U8);
    let s32 = ElemType::Int(IntKind::I32);

    let config = |a_type, b_type, cd_type, k, n_granularity| WgmmaConfig {
        a_type,
        b_type,
        cd_type,
        m: M,
        n_granularity,
        n_max: N_MAX,
        k,
    };

    let mut configs = vec![
        config(f16, f16, f16, 16, 8),
        config(f16, f16, f32, 16, 8),
        config(bf16, bf16, f32, 16, 8),
        config(tf32, tf32, f32, 8, 8),
    ];
    for a in [e4m3, e5m2] {
        for b in [e4m3, e5m2] {
            configs.push(config(a, b, f16, 32, 8));
            configs.push(config(a, b, f32, 32, 8));
        }
    }
    // Integer forms also take n = 8 and 24, but every other size is a multiple of 16.
    for a in [s8, u8] {
        for b in [s8, u8] {
            configs.push(config(a, b, s32, 32, 16));
        }
    }
    configs
}

/// The PTX name of a warpgroup MMA element.
fn ptx_type(elem: ElemType) -> &'static str {
    match elem {
        ElemType::Float(FloatKind::F16) => "f16",
        ElemType::Float(FloatKind::BF16) => "bf16",
        ElemType::Float(FloatKind::TF32) => "tf32",
        ElemType::Float(FloatKind::E4M3) => "e4m3",
        ElemType::Float(FloatKind::E5M2) => "e5m2",
        ElemType::Float(FloatKind::F32) => "f32",
        ElemType::Int(IntKind::I8) => "s8",
        ElemType::UInt(UIntKind::U8) => "u8",
        ElemType::Int(IntKind::I32) => "s32",
        other => unreachable!("{other:?} is in no warpgroup MMA configuration"),
    }
}

fn elem_type(ctx: &Context, ty: TypeHandle) -> ElemType {
    type_cast::<dyn ScalarType>(&*ty.deref(ctx))
        .expect("a matrix element is a scalar")
        .elem_type(ctx)
}

fn is_half(elem: ElemType) -> bool {
    matches!(
        elem,
        ElemType::Float(FloatKind::F16) | ElemType::Float(FloatKind::BF16)
    )
}

fn is_integer(elem: ElemType) -> bool {
    matches!(elem, ElemType::Int(_) | ElemType::UInt(_))
}

/// How an inline assembly operand list takes registers of `elem`: one per 32-bit element, or
/// the elements packed into 32-bit words.
fn register_form(elem: ElemType) -> RegisterForm {
    if elem.size_bits() == 32 {
        RegisterForm::Scalar
    } else {
        RegisterForm::Word
    }
}

/// The inline assembly constraint of a 32-bit register.
fn constraint(ctx: &Context, reg: Value) -> &'static str {
    if reg.get_type(ctx).deref(ctx).is::<FP32Type>() {
        "f"
    } else {
        "r"
    }
}

/// Calls an inline assembly `template` that reads and writes `regs` in place, along with
/// `inputs`, and returns the registers it wrote.
fn asm_in_place(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    template: &str,
    regs: Vec<Value>,
    inputs: Vec<(Value, &'static str)>,
    clobbers_memory: bool,
    convergent: bool,
) -> Vec<Value> {
    let count = regs.len();
    let reg_tys: Vec<TypeHandle> = regs.iter().map(|reg| reg.get_type(ctx)).collect();

    let mut constraints: Vec<String> = regs
        .iter()
        .map(|&reg| format!("={}", constraint(ctx, reg)))
        .collect();
    constraints.extend((0..count).map(|i| i.to_string()));
    constraints.extend(inputs.iter().map(|(_, c)| c.to_string()));
    if clobbers_memory {
        constraints.push("~{memory}".into());
    }

    let mut args = regs;
    args.extend(inputs.into_iter().map(|(value, _)| value));

    let result_ty = registers_result_ty(ctx, reg_tys);
    let asm = llvm::InlineAsmOp::new(ctx, result_ty, args, template, &constraints.join(","), true);
    if convergent {
        let mut attrs = LlvmAttributesAttr::new();
        attrs.set("convergent", LlvmAttrValue::Unit);
        asm.set_attr_llvm_inline_asm_attrs(ctx, attrs);
    }
    let result = insert(ctx, rw, &asm);
    unpack_registers(ctx, rw, result, count)
}

/// A run of `$i` operand references, braced as a PTX vector operand.
fn operand_list(first: usize, count: usize) -> String {
    let operands: Vec<String> = (first..first + count).map(|i| format!("${i}")).collect();
    format!("{{{}}}", operands.join(", "))
}

fn int_width(ctx: &Context, value: Value) -> Option<u32> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int| int.width())
}

pub(crate) fn wgmma(
    op: &WgmmaOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let (a, b, acc, scale_d) = (op.a(ctx), op.b(ctx), op.accumulator(ctx), op.scale_d(ctx));
    // The operands are already converted: a descriptor is an integer, registers an array.
    let a_in_registers = int_width(ctx, a).is_none();
    let a_ty = op.a_ty(ctx).get_type(ctx);
    let b_ty = op.b_ty(ctx).get_type(ctx);
    let a_elem = elem_type(ctx, a_ty);
    let b_elem = elem_type(ctx, b_ty);
    let shape = *op.shape(ctx).clone();
    let a_layout = op.a_layout(ctx).0;
    let b_layout = op.b_layout(ctx).0;

    let (acc_vec_ty, cd_scalar) = registers_as_vector(ctx, operands_info, acc);
    let cd_elem = elem_type(ctx, cd_scalar);

    let supported = wgmma_configs();
    let MatrixShape { m, n, k } = shape;
    let valid = supported
        .iter()
        .any(|config| config.matches(a_elem, b_elem, cd_elem, m as u32, n as u32, k as u32));
    if !valid {
        return input_err!(
            op.loc(ctx),
            WgmmaUnsupported {
                a: a_elem,
                b: b_elem,
                cd: cd_elem,
                shape,
                supported: supported.to_vec(),
            }
        );
    }

    // Only 16-bit tiles may be MN-major; the others must be K-major, which is a row-major `A`
    // and a column-major `B`.
    let half = is_half(a_elem);
    let trans_a = !a_in_registers && a_layout == MatrixLayout::ColMajor;
    let trans_b = b_layout == MatrixLayout::RowMajor;
    if !half && trans_a {
        return input_err!(op.loc(ctx), WgmmaLayoutUnsupported("A", a_elem));
    }
    if !half && trans_b {
        return input_err!(op.loc(ctx), WgmmaLayoutUnsupported("B", b_elem));
    }

    let acc_elems = vector_lanes(ctx, acc_vec_ty);
    let expected_acc = (M as usize * shape.n) / WARPGROUP_UNITS as usize;
    if acc_elems != expected_acc {
        return input_err!(
            op.loc(ctx),
            WgmmaRegisterCount {
                what: "accumulator",
                unit: "elements",
                found: acc_elems,
                expected: expected_acc,
            }
        );
    }

    let acc_form = register_form(cd_elem);
    let acc_value = registers_value(ctx, rw, acc, acc_vec_ty);
    let acc_regs = registers_of(ctx, rw, acc_value, acc_form);
    let count = acc_regs.len();

    // Inputs follow the accumulator's outputs and the inputs tied to them.
    let mut inputs = vec![];
    let a_operand = if a_in_registers {
        let (a_vec_ty, _) = registers_as_vector(ctx, operands_info, a);
        let bits = vector_bits(ctx, a_vec_ty);
        if bits != A_REGISTERS * 32 {
            return input_err!(
                op.loc(ctx),
                WgmmaRegisterCount {
                    what: "A fragment",
                    unit: "bits",
                    found: bits,
                    expected: A_REGISTERS * 32,
                }
            );
        }
        let a_value = registers_value(ctx, rw, a, a_vec_ty);
        let a_regs = registers_of(ctx, rw, a_value, RegisterForm::Word);
        let operand = operand_list(2 * count, a_regs.len());
        inputs.extend(a_regs.into_iter().map(|reg| (reg, "r")));
        operand
    } else {
        if int_width(ctx, a) != Some(64) {
            let ty = a.get_type(ctx).disp(ctx).to_string();
            return input_err!(op.loc(ctx), WgmmaDescriptorType(ty));
        }
        inputs.push((a, "l"));
        format!("${}", 2 * count)
    };
    if int_width(ctx, b) != Some(64) {
        let ty = b.get_type(ctx).disp(ctx).to_string();
        return input_err!(op.loc(ctx), WgmmaDescriptorType(ty));
    }
    let b_operand = format!("${}", 2 * count + inputs.len());
    inputs.push((b, "l"));
    let scale_operand = format!("${}", 2 * count + inputs.len());
    let scale_d = resize_int(ctx, rw, scale_d, 32, false).expect("`scale_d` is a boolean");
    inputs.push((scale_d, "r"));

    let mut instruction = format!(
        "wgmma.mma_async.sync.aligned.m{M}n{n}k{k}.{}.{}.{} {}, {a_operand}, {b_operand}, p",
        ptx_type(cd_elem),
        ptx_type(a_elem),
        ptx_type(b_elem),
        operand_list(0, count),
    );
    // Integer forms take neither the operand negation nor the transposes.
    if !is_integer(a_elem) {
        instruction.push_str(", 1, 1");
    }
    if half {
        if !a_in_registers {
            instruction.push_str(&format!(", {}", trans_a as u32));
        }
        instruction.push_str(&format!(", {}", trans_b as u32));
    }
    let template =
        format!("{{\n.reg .pred p;\nsetp.ne.b32 p, {scale_operand}, 0;\n{instruction};\n}}");

    let regs = asm_in_place(ctx, rw, &template, acc_regs, inputs, true, true);

    let result = registers_into(ctx, rw, &regs, acc_vec_ty, acc_form);
    let acc_array_ty = registers_array_ty(ctx, operands_info, acc);
    let result = vector_into_array(ctx, rw, result, acc_array_ty);
    store_fragment(ctx, rw, acc, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// An empty inline assembly that reads and writes the registers: no access to them moves across
/// it, and they are not copied out of the registers an MMA in flight writes.
pub(crate) fn fence_operand(
    op: &WgmmaFenceOperandOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let registers = op.registers(ctx);

    let (vec_ty, scalar) = registers_as_vector(ctx, operands_info, registers);
    let form = register_form(elem_type(ctx, scalar));
    let bits = vector_bits(ctx, vec_ty);
    if form == RegisterForm::Word && !bits.is_multiple_of(32) {
        return input_err!(
            op.loc(ctx),
            WgmmaRegisterCount {
                what: "fenced array",
                unit: "bits",
                found: bits,
                expected: bits.next_multiple_of(32),
            }
        );
    }

    let value = registers_value(ctx, rw, registers, vec_ty);
    let regs = registers_of(ctx, rw, value, form);
    let regs = asm_in_place(ctx, rw, "", regs, vec![], false, false);

    let result = registers_into(ctx, rw, &regs, vec_ty, form);
    let array_ty = registers_array_ty(ctx, operands_info, registers);
    let result = vector_into_array(ctx, rw, result, array_ty);
    store_fragment(ctx, rw, registers, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// The descriptor's address and offset fields each hold bits 4 to 17 of a byte value.
fn descriptor_field(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    shift: i128,
) -> Value {
    let value = resize_int(ctx, rw, value, 64, false).expect("descriptor fields are integers");
    let four = insert_int_const(ctx, rw, 64, 4);
    let op = llvm::LShrOp::new(ctx, value, four);
    let value = insert(ctx, rw, &op);
    let mask = insert_int_const(ctx, rw, 64, 0x3FFF);
    let op = llvm::AndOp::new(ctx, value, mask);
    let value = insert(ctx, rw, &op);
    if shift == 0 {
        return value;
    }
    let shift = insert_int_const(ctx, rw, 64, shift);
    let op =
        llvm::ShlOp::new_with_overflow_flag(ctx, value, shift, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

pub(crate) fn descriptor(
    op: &WgmmaDescriptorOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let ptr = op.ptr(ctx);
    let leading = op.leading_byte_offset(ctx);
    let stride = op.stride_byte_offset(ctx);
    let swizzle = op.swizzle(ctx).descriptor_bits();

    let shared = as_shared(ctx, rw, ptr);
    let i64_ty = i64_ty(ctx);
    let op_addr = llvm::PtrToIntOp::new(ctx, shared, i64_ty);
    let address = insert(ctx, rw, &op_addr);

    let mut descriptor = descriptor_field(ctx, rw, address, 0);
    for (value, shift) in [(leading, 16), (stride, 32)] {
        let field = descriptor_field(ctx, rw, value, shift);
        let op = llvm::OrOp::new(ctx, descriptor, field);
        descriptor = insert(ctx, rw, &op);
    }
    if swizzle != 0 {
        let bits = insert_int_const(ctx, rw, 64, (swizzle << 62) as i64 as i128);
        let op = llvm::OrOp::new(ctx, descriptor, bits);
        descriptor = insert(ctx, rw, &op);
    }

    rw.replace_operation_with_values(ctx, old_op, vec![descriptor]);
    Ok(())
}

pub(crate) fn fence(
    op: &WgmmaFenceOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    call_void(ctx, rw, FENCE, vec![]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn commit_group(
    op: &WgmmaCommitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    call_void(ctx, rw, COMMIT_GROUP, vec![]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_group(
    op: &WgmmaWaitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let max_pending = op.max_pending(ctx).0;
    let max_pending = insert_int_const(ctx, rw, 64, max_pending as i128);
    call_void(ctx, rw, WAIT_GROUP, vec![max_pending]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

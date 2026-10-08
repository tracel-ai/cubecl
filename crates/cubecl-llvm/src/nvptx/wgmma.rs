//! NVPTX warpgroup MMA, Hopper's asynchronous tensor core instructions.
//!
//! LLVM has intrinsics for the fence and for committing and waiting on groups, but none for
//! `wgmma.mma_async` itself, so the MMA is inline PTX, as MLIR's NVVM dialect emits it.

use crate::{
    nvptx::{
        address::NvptxSpace,
        inline_asm::InlineAsm,
        registers::{RegisterForm, store_fragment, vector_bits, vector_lanes},
    },
    prelude::*,
    shared::matrix::{registers_array_ty, registers_as_vector, registers_value, vector_into_array},
};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    dialect::matrix::{
        WgmmaCommitGroupOp, WgmmaDescriptorOp, WgmmaFenceOp, WgmmaFenceOperandOp, WgmmaMajor,
        WgmmaOp, WgmmaSwizzle, WgmmaWaitGroupOp,
    },
    features::{WgmmaConfig, WgmmaElems},
    types::{MatrixIdent, MatrixShape},
};
use pliron::r#type::type_cast;
use std::sync::LazyLock;

const FENCE: &str = "llvm.nvvm.wgmma.fence.sync.aligned";
const COMMIT_GROUP: &str = "llvm.nvvm.wgmma.commit_group.sync.aligned";
const WAIT_GROUP: &str = "llvm.nvvm.wgmma.wait_group.sync.aligned";

/// Units of the four planes that issue one warpgroup MMA together.
const WARPGROUP_UNITS: usize = 128;
/// Every warpgroup MMA computes 64 rows.
const M: usize = 64;
const N_MAX: u32 = 256;
/// `A` held in registers is always four 32-bit registers per unit.
const A_REGISTERS: usize = 4;

/// A warpgroup MMA no PTX form takes.
#[derive(Debug, Error)]
pub enum WgmmaError {
    #[error(
        "no warpgroup MMA multiplies {:?} by {:?} into {:?} with shape {shape:?}; supported \
         configurations: {supported:?}",
        elems.a, elems.b, elems.cd
    )]
    Unsupported {
        elems: WgmmaElems,
        shape: MatrixShape,
        supported: Vec<WgmmaConfig>,
    },
    #[error(
        "a warpgroup MMA reads {operand} only K-major for {elem:?}: MN-major tiles are for \
         16-bit elements"
    )]
    MnMajor {
        operand: MatrixIdent,
        elem: ElemType,
    },
    #[error(
        "the warpgroup MMA {operand} holds {found} {unit} per unit, and the instruction takes {expected}"
    )]
    RegisterCount {
        operand: MatrixIdent,
        unit: &'static str,
        found: usize,
        expected: usize,
    },
    #[error("a warpgroup MMA matrix descriptor is a u64, not {0}")]
    DescriptorType(String),
}

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
        m: M as u32,
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

/// Where a warpgroup MMA reads `A` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgmmaA {
    /// A shared memory tile, through a descriptor.
    Shared(WgmmaMajor),
    /// Four 32-bit registers per unit.
    Registers,
}

/// A warpgroup MMA PTX has a form for: the element types, the shape, and how the operands are
/// read. Its [`Display`](core::fmt::Display) is the instruction, without operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgmmaForm {
    elems: WgmmaElems,
    n: usize,
    k: usize,
    a: WgmmaA,
    b_major: WgmmaMajor,
}

impl WgmmaForm {
    pub fn new(
        elems: WgmmaElems,
        shape: MatrixShape,
        a: WgmmaA,
        b_major: WgmmaMajor,
    ) -> core::result::Result<Self, WgmmaError> {
        let supported = wgmma_configs();
        if !supported.iter().any(|config| config.matches(elems, shape)) {
            return Err(WgmmaError::Unsupported {
                elems,
                shape,
                supported: supported.to_vec(),
            });
        }
        let half = is_half(elems.a);
        if !half && a == WgmmaA::Shared(WgmmaMajor::MN) {
            return Err(WgmmaError::MnMajor {
                operand: MatrixIdent::A,
                elem: elems.a,
            });
        }
        if !half && b_major == WgmmaMajor::MN {
            return Err(WgmmaError::MnMajor {
                operand: MatrixIdent::B,
                elem: elems.b,
            });
        }
        Ok(Self {
            elems,
            n: shape.n,
            k: shape.k,
            a,
            b_major,
        })
    }

    /// The accumulator elements each unit holds.
    pub fn accumulator_elems(&self) -> usize {
        M * self.n / WARPGROUP_UNITS
    }

    /// The immediate operands that follow `scale-d`: the operand negations, which integer forms
    /// lack, then the transposes, which only 16-bit forms take, and only of a tile.
    pub fn immediates(&self) -> String {
        let mut immediates = String::new();
        if !is_integer(self.elems.a) {
            immediates.push_str(", 1, 1");
        }
        if is_half(self.elems.a) {
            if let WgmmaA::Shared(major) = self.a {
                immediates.push_str(&format!(", {}", transposed(major)));
            }
            immediates.push_str(&format!(", {}", transposed(self.b_major)));
        }
        immediates
    }
}

impl core::fmt::Display for WgmmaForm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let WgmmaElems { a, b, cd } = self.elems;
        write!(
            f,
            "wgmma.mma_async.sync.aligned.m{M}n{}k{}.{}.{}.{}",
            self.n,
            self.k,
            ptx_type(cd),
            ptx_type(a),
            ptx_type(b)
        )
    }
}

/// A K-major tile is the one PTX leaves untransposed.
fn transposed(major: WgmmaMajor) -> u32 {
    match major {
        WgmmaMajor::K => 0,
        WgmmaMajor::MN => 1,
    }
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

fn int_width(ctx: &Context, value: Value) -> Option<u32> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int| int.width())
}

/// `value` as a descriptor operand of `asm`.
fn descriptor_operand(
    ctx: &Context,
    asm: &mut InlineAsm,
    value: Value,
) -> core::result::Result<String, WgmmaError> {
    if int_width(ctx, value) != Some(64) {
        let ty = value.get_type(ctx).disp(ctx).to_string();
        return Err(WgmmaError::DescriptorType(ty));
    }
    Ok(asm.input(value, "l"))
}

pub(crate) fn wgmma(
    op: &WgmmaOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    match issue(op, ctx, rw, operands_info) {
        Ok(()) => Ok(()),
        Err(err) => input_err!(loc, err),
    }
}

fn issue(
    op: &WgmmaOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> core::result::Result<(), WgmmaError> {
    let old_op = op.get_operation();
    let (a, b, acc) = (op.a(ctx), op.b(ctx), op.accumulator(ctx));
    // The operands are already converted: a descriptor is an integer, registers an array.
    let a_source = match int_width(ctx, a) {
        Some(_) => WgmmaA::Shared(*op.a_major(ctx)),
        None => WgmmaA::Registers,
    };
    let (acc_vec_ty, cd_scalar) = registers_as_vector(ctx, operands_info, acc);
    let elems = WgmmaElems {
        a: elem_type(ctx, op.a_ty(ctx).get_type(ctx)),
        b: elem_type(ctx, op.b_ty(ctx).get_type(ctx)),
        cd: elem_type(ctx, cd_scalar),
    };
    let shape = *op.shape(ctx).clone();
    let form = WgmmaForm::new(elems, shape, a_source, *op.b_major(ctx))?;

    let acc_elems = vector_lanes(ctx, acc_vec_ty);
    if acc_elems != form.accumulator_elems() {
        return Err(WgmmaError::RegisterCount {
            operand: MatrixIdent::Accumulator,
            unit: "elements",
            found: acc_elems,
            expected: form.accumulator_elems(),
        });
    }

    let acc_form = register_form(elems.cd);
    let acc_value = registers_value(ctx, rw, acc, acc_vec_ty);
    let acc_regs = acc_form.split(ctx, rw, acc_value);

    let mut asm = InlineAsm::new()
        .tied(acc_regs)
        .clobbers_memory()
        .convergent();
    let a_operand = match a_source {
        WgmmaA::Shared(_) => descriptor_operand(ctx, &mut asm, a)?,
        WgmmaA::Registers => {
            let (a_vec_ty, _) = registers_as_vector(ctx, operands_info, a);
            let bits = vector_bits(ctx, a_vec_ty);
            if bits != A_REGISTERS * 32 {
                return Err(WgmmaError::RegisterCount {
                    operand: MatrixIdent::A,
                    unit: "bits",
                    found: bits,
                    expected: A_REGISTERS * 32,
                });
            }
            let a_value = registers_value(ctx, rw, a, a_vec_ty);
            let a_regs = RegisterForm::Word.split(ctx, rw, a_value);
            let operands: Vec<String> = a_regs.into_iter().map(|reg| asm.input(reg, "r")).collect();
            format!("{{{}}}", operands.join(", "))
        }
    };
    let b_operand = descriptor_operand(ctx, &mut asm, b)?;
    // The accumulator always adds to what it holds: a new one is zeroed.
    let accumulate = insert_int_const(ctx, rw, 32, 1);
    let scale_d = asm.input(accumulate, "r");

    let template = format!(
        "{{\n.reg .pred p;\nsetp.ne.b32 p, {scale_d}, 0;\n{form} {}, {a_operand}, {b_operand}, p{};\n}}",
        asm.tied_operands(),
        form.immediates(),
    );
    let regs = asm.emit(ctx, rw, &template);

    let result = acc_form.join(ctx, rw, &regs, acc_vec_ty);
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
            WgmmaError::RegisterCount {
                operand: MatrixIdent::Accumulator,
                unit: "bits",
                found: bits,
                expected: bits.next_multiple_of(32),
            }
        );
    }

    let value = registers_value(ctx, rw, registers, vec_ty);
    let regs = form.split(ctx, rw, value);
    let regs = InlineAsm::new().tied(regs).emit(ctx, rw, "");

    let result = form.join(ctx, rw, &regs, vec_ty);
    let array_ty = registers_array_ty(ctx, operands_info, registers);
    let result = vector_into_array(ctx, rw, result, array_ty);
    store_fragment(ctx, rw, registers, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// The descriptor's fields that don't depend on the address: the offsets, already in 16-byte
/// units, and the swizzle.
fn descriptor_constant_bits(leading: usize, stride: usize, swizzle: WgmmaSwizzle) -> u64 {
    let field = |bytes: usize| (bytes as u64 >> 4) & 0x3FFF;
    field(leading) << 16 | field(stride) << 32 | swizzle.descriptor_bits() << 62
}

pub(crate) fn descriptor(
    op: &WgmmaDescriptorOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let ptr = op.ptr(ctx);
    let constant = descriptor_constant_bits(
        op.leading_byte_offset(ctx).0,
        op.stride_byte_offset(ctx).0,
        *op.swizzle(ctx),
    );

    let shared = NvptxSpace::Shared.cast(ctx, rw, ptr);
    let i64_ty = i64_ty(ctx);
    let op_addr = llvm::PtrToIntOp::new(ctx, shared, i64_ty);
    let address = insert(ctx, rw, &op_addr);
    // The address field holds bits 4 to 17 of the shared memory address.
    let four = insert_int_const(ctx, rw, 64, 4);
    let op = llvm::LShrOp::new(ctx, address, four);
    let address = insert(ctx, rw, &op);
    let mask = insert_int_const(ctx, rw, 64, 0x3FFF);
    let op = llvm::AndOp::new(ctx, address, mask);
    let address = insert(ctx, rw, &op);
    let constant = insert_int_const(ctx, rw, 64, constant as i64 as i128);
    let op = llvm::OrOp::new(ctx, address, constant);
    let descriptor = insert(ctx, rw, &op);

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

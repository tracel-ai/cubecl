//! NVPTX warpgroup MMA, Hopper's asynchronous tensor core instructions.
//!
//! LLVM has intrinsics for the fence and for committing and waiting on groups, but none for
//! `wgmma.mma_async` itself, so the MMA is inline PTX, as MLIR's NVVM dialect emits it.

use crate::{
    nvptx::{
        address::NvptxSpace,
        inline_asm::InlineAsm,
        registers::{RegisterForm, RegisterOperand},
    },
    prelude::*,
};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    dialect::matrix::{
        WARPGROUP_M, WgmmaCommitGroupOp, WgmmaDescriptorOp, WgmmaFenceOp, WgmmaFenceOperandOp,
        WgmmaMajor, WgmmaOp, WgmmaSwizzle, WgmmaWaitGroupOp, warpgroup_elems_per_unit,
    },
    features::{WgmmaConfig, WgmmaElems},
    nvidia::SmArch,
    types::{MatrixIdent, MatrixShape},
};
use pliron::r#type::type_cast;
use std::sync::LazyLock;

const FENCE: &str = "llvm.nvvm.wgmma.fence.sync.aligned";
const COMMIT_GROUP: &str = "llvm.nvvm.wgmma.commit_group.sync.aligned";
const WAIT_GROUP: &str = "llvm.nvvm.wgmma.wait_group.sync.aligned";

const N_MAX: u32 = 256;
/// Every warpgroup MMA reads 32 bytes of K.
const K_BYTES: usize = 32;
/// `A` held in registers is always four 32-bit registers per unit.
const A_REGISTER_BITS: usize = 4 * 32;
/// A descriptor holds its address and offsets in 14-bit fields of 16-byte units.
const DESCRIPTOR_UNIT: usize = 16;
const DESCRIPTOR_FIELD_MASK: u64 = 0x3FFF;

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
        "the warpgroup MMA accumulator holds {found} elements per unit, and the instruction \
         takes {expected}"
    )]
    AccumulatorElems { found: usize, expected: usize },
    #[error(
        "the warpgroup MMA `A` fragment holds {found} bits per unit, and the instruction takes \
         {A_REGISTER_BITS}"
    )]
    ARegisterBits { found: usize },
    #[error("registers pinned for a warpgroup MMA fill whole 32-bit registers, not {bits} bits")]
    UnalignedRegisters { bits: usize },
    #[error(
        "a warpgroup MMA descriptor offset is a multiple of 16 bytes under 256 KiB, not {bytes}"
    )]
    DescriptorOffset { bytes: usize },
    #[error("a warpgroup MMA matrix descriptor is a u64, not {0}")]
    DescriptorType(String),
}

/// The warpgroup MMA forms `arch` has: Hopper's alone, since Blackwell replaced them with
/// `tcgen05`.
pub fn configs(arch: SmArch) -> &'static [WgmmaConfig] {
    static HOPPER: LazyLock<Vec<WgmmaConfig>> = LazyLock::new(hopper_configs);
    match arch.version() {
        90 => &HOPPER,
        _ => &[],
    }
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

    // Every form reads 32 bytes of K, and takes any `n` that is a multiple of 8.
    let config = |a: ElemType, b, cd| WgmmaConfig {
        elems: WgmmaElems { a, b, cd },
        m: WARPGROUP_M as u32,
        n_granularity: 8,
        n_max: N_MAX,
        k: (K_BYTES / a.size()) as u32,
    };

    let mut configs = vec![
        config(f16, f16, f16),
        config(f16, f16, f32),
        config(bf16, bf16, f32),
        config(tf32, tf32, f32),
    ];
    for a in [e4m3, e5m2] {
        for b in [e4m3, e5m2] {
            configs.push(config(a, b, f16));
            configs.push(config(a, b, f32));
        }
    }
    // Integer forms also take n = 8 and 24, but every other size is a multiple of 16.
    for a in [s8, u8] {
        for b in [s8, u8] {
            configs.push(WgmmaConfig {
                n_granularity: 16,
                ..config(a, b, s32)
            });
        }
    }
    configs
}

/// Where a warpgroup MMA reads `A` from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgmmaASource {
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
    a: WgmmaASource,
    b_major: WgmmaMajor,
}

impl WgmmaForm {
    pub fn new(
        elems: WgmmaElems,
        shape: MatrixShape,
        a: WgmmaASource,
        b_major: WgmmaMajor,
    ) -> core::result::Result<Self, WgmmaError> {
        let supported = &HOPPER_FORMS;
        if !supported.iter().any(|config| config.matches(elems, shape)) {
            return Err(WgmmaError::Unsupported {
                elems,
                shape,
                supported: supported.to_vec(),
            });
        }
        let half = is_half(elems.a);
        if !half && a == WgmmaASource::Shared(WgmmaMajor::MN) {
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

    /// Checks that the accumulator holds the elements the instruction writes.
    pub fn check_accumulator(&self, elems: usize) -> core::result::Result<(), WgmmaError> {
        let expected = warpgroup_elems_per_unit(self.n);
        if elems != expected {
            return Err(WgmmaError::AccumulatorElems {
                found: elems,
                expected,
            });
        }
        Ok(())
    }

    /// Checks that `A` in registers fills the four registers the instruction reads.
    pub fn check_a_registers(&self, bits: usize) -> core::result::Result<(), WgmmaError> {
        if bits != A_REGISTER_BITS {
            return Err(WgmmaError::ARegisterBits { found: bits });
        }
        Ok(())
    }

    /// How the accumulator's elements go into the instruction's registers: one per 32-bit
    /// element, or packed into 32-bit words.
    pub(crate) fn accumulator_form(&self) -> RegisterForm {
        if self.elems.cd.size_bits() == 32 {
            RegisterForm::Scalar
        } else {
            RegisterForm::Word
        }
    }

    /// The immediate operands that follow `scale-d`: the operand negations, which integer forms
    /// lack, then the transposes, which only 16-bit forms take, and only of a tile.
    pub fn immediates(&self) -> String {
        let mut immediates = String::new();
        if !is_integer(self.elems.a) {
            immediates.push_str(", 1, 1");
        }
        if is_half(self.elems.a) {
            if let WgmmaASource::Shared(major) = self.a {
                immediates.push_str(&format!(", {}", transposed(major)));
            }
            immediates.push_str(&format!(", {}", transposed(self.b_major)));
        }
        immediates
    }
}

/// Every form [`WgmmaForm`] spells, whatever the device: the lowering checks the instruction,
/// and the device properties decide what a kernel may use.
static HOPPER_FORMS: LazyLock<Vec<WgmmaConfig>> = LazyLock::new(hopper_configs);

impl core::fmt::Display for WgmmaForm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let WgmmaElems { a, b, cd } = self.elems;
        write!(
            f,
            "wgmma.mma_async.sync.aligned.m{WARPGROUP_M}n{}k{}.{}.{}.{}",
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

/// `value` as a descriptor operand of `asm`.
fn descriptor_operand(
    ctx: &Context,
    asm: &mut InlineAsm,
    value: Value,
) -> core::result::Result<String, WgmmaError> {
    let is_u64 = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int| int.width() == 64);
    if !is_u64 {
        let ty = value.get_type(ctx).disp(ctx).to_string();
        return Err(WgmmaError::DescriptorType(ty));
    }
    Ok(asm.input(value))
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
    // A descriptor is an integer before conversion, and registers an array.
    let a_is_descriptor = cube_origin(ctx, operands_info, a, |ty| {
        Some(type_cast::<dyn ScalarType>(ty).is_some())
    })
    .unwrap_or(false);
    let a_source = match a_is_descriptor {
        true => WgmmaASource::Shared(*op.a_major(ctx)),
        false => WgmmaASource::Registers,
    };
    let elems = WgmmaElems {
        a: elem_type(ctx, op.a_ty(ctx).get_type(ctx)),
        b: elem_type(ctx, op.b_ty(ctx).get_type(ctx)),
        cd: {
            let acc = RegisterOperand::new(ctx, operands_info, acc, |_, _| RegisterForm::Scalar);
            elem_type(ctx, acc.elem())
        },
    };
    let shape = *op.shape(ctx).clone();
    let form = WgmmaForm::new(elems, shape, a_source, *op.b_major(ctx))?;

    let acc = RegisterOperand::new(ctx, operands_info, acc, |_, _| form.accumulator_form());
    form.check_accumulator(acc.lanes(ctx))?;
    let acc_regs = acc.load(ctx, rw);

    let mut asm = InlineAsm::new(acc_regs).clobbers_memory().convergent();
    let a_operand = match a_source {
        WgmmaASource::Shared(_) => descriptor_operand(ctx, &mut asm, a)?,
        WgmmaASource::Registers => {
            let fragment = RegisterOperand::new(ctx, operands_info, a, |_, _| RegisterForm::Word);
            form.check_a_registers(fragment.bits(ctx))?;
            let operands: Vec<String> = fragment
                .load(ctx, rw)
                .into_iter()
                .map(|reg| asm.input(reg))
                .collect();
            format!("{{{}}}", operands.join(", "))
        }
    };
    let b_operand = descriptor_operand(ctx, &mut asm, b)?;
    // The accumulator always adds to what it holds: a new one is zeroed.
    let accumulate = insert_int_const(ctx, rw, 32, 1);
    let scale_d = asm.input(accumulate);

    let template = format!(
        "{{\n.reg .pred p;\nsetp.ne.b32 p, {scale_d}, 0;\n{form} {}, {a_operand}, {b_operand}, p{};\n}}",
        asm.tied_operands(),
        form.immediates(),
    );
    let regs = asm.emit(ctx, rw, &template);
    acc.store(ctx, rw, operands_info, &regs);

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
    let registers = RegisterOperand::new(ctx, operands_info, op.registers(ctx), |ctx, elem| {
        if elem_type(ctx, elem).size_bits() == 32 {
            RegisterForm::Scalar
        } else {
            RegisterForm::Word
        }
    });
    let bits = registers.bits(ctx);
    if !bits.is_multiple_of(32) {
        return input_err!(op.loc(ctx), WgmmaError::UnalignedRegisters { bits });
    }

    let regs = registers.load(ctx, rw);
    let regs = InlineAsm::new(regs).emit(ctx, rw, "");
    registers.store(ctx, rw, operands_info, &regs);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// The value of the descriptor's two-bit swizzle field.
fn swizzle_field(swizzle: WgmmaSwizzle) -> u64 {
    match swizzle {
        WgmmaSwizzle::None => 0,
        WgmmaSwizzle::B128 => 1,
        WgmmaSwizzle::B64 => 2,
        WgmmaSwizzle::B32 => 3,
    }
}

/// The descriptor field holding `bytes`, which is a multiple of 16 that fits in 14 bits once
/// divided by 16.
fn descriptor_field(bytes: usize) -> core::result::Result<u64, WgmmaError> {
    let units = (bytes / DESCRIPTOR_UNIT) as u64;
    if !bytes.is_multiple_of(DESCRIPTOR_UNIT) || units > DESCRIPTOR_FIELD_MASK {
        return Err(WgmmaError::DescriptorOffset { bytes });
    }
    Ok(units)
}

/// The descriptor's fields that don't depend on the address: the offsets, given in bytes, and
/// the swizzle.
fn descriptor_constant_bits(
    leading_bytes: usize,
    stride_bytes: usize,
    swizzle: WgmmaSwizzle,
) -> core::result::Result<u64, WgmmaError> {
    let leading = descriptor_field(leading_bytes)?;
    let stride = descriptor_field(stride_bytes)?;
    Ok(leading << 16 | stride << 32 | swizzle_field(swizzle) << 62)
}

pub(crate) fn descriptor(
    op: &WgmmaDescriptorOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let ptr = op.ptr(ctx);
    let constant = match descriptor_constant_bits(
        op.leading_byte_offset(ctx).0,
        op.stride_byte_offset(ctx).0,
        *op.swizzle(ctx),
    ) {
        Ok(constant) => constant,
        Err(err) => return input_err!(op.loc(ctx), err),
    };

    let shared = NvptxSpace::Shared.cast(ctx, rw, ptr);
    let i64_ty = i64_ty(ctx);
    let op_addr = llvm::PtrToIntOp::new(ctx, shared, i64_ty);
    let address = insert(ctx, rw, &op_addr);
    // The address field holds bits 4 to 17 of the shared memory address.
    let four = insert_int_const(ctx, rw, 64, 4);
    let op = llvm::LShrOp::new(ctx, address, four);
    let address = insert(ctx, rw, &op);
    let mask = insert_int_const(ctx, rw, 64, DESCRIPTOR_FIELD_MASK as i128);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn elems(a: ElemType, b: ElemType, cd: ElemType) -> WgmmaElems {
        WgmmaElems { a, b, cd }
    }

    fn shape(n: usize, k: usize) -> MatrixShape {
        MatrixShape {
            m: WARPGROUP_M,
            n,
            k,
        }
    }

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const TF32: ElemType = ElemType::Float(FloatKind::TF32);
    const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
    const S8: ElemType = ElemType::Int(IntKind::I8);
    const S32: ElemType = ElemType::Int(IntKind::I32);

    fn spelled(form: WgmmaForm) -> String {
        format!("{form}{}", form.immediates())
    }

    #[test]
    fn half_forms_name_both_transposes_of_tiles() {
        let a = WgmmaASource::Shared(WgmmaMajor::MN);
        let form = WgmmaForm::new(elems(F16, F16, F32), shape(64, 16), a, WgmmaMajor::K);
        assert_eq!(
            spelled(form.unwrap()),
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16, 1, 1, 1, 0"
        );
    }

    #[test]
    fn half_forms_name_only_bs_transpose_with_a_in_registers() {
        let a = WgmmaASource::Registers;
        let form = WgmmaForm::new(elems(F16, F16, F16), shape(8, 16), a, WgmmaMajor::MN);
        assert_eq!(
            spelled(form.unwrap()),
            "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16, 1, 1, 1"
        );
    }

    #[test]
    fn tf32_and_fp8_forms_take_negations_only() {
        let a = WgmmaASource::Shared(WgmmaMajor::K);
        let tf32 = WgmmaForm::new(elems(TF32, TF32, F32), shape(256, 8), a, WgmmaMajor::K);
        assert_eq!(
            spelled(tf32.unwrap()),
            "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32, 1, 1"
        );
        let fp8 = WgmmaForm::new(elems(E4M3, E4M3, F16), shape(64, 32), a, WgmmaMajor::K);
        assert_eq!(
            spelled(fp8.unwrap()),
            "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3, 1, 1"
        );
    }

    #[test]
    fn integer_forms_take_no_immediates() {
        let a = WgmmaASource::Shared(WgmmaMajor::K);
        let form = WgmmaForm::new(elems(S8, S8, S32), shape(32, 32), a, WgmmaMajor::K);
        assert_eq!(
            spelled(form.unwrap()),
            "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.s8"
        );
    }

    #[test]
    fn only_half_tiles_may_be_mn_major() {
        let mn = WgmmaASource::Shared(WgmmaMajor::MN);
        let tf32 = WgmmaForm::new(elems(TF32, TF32, F32), shape(64, 8), mn, WgmmaMajor::K);
        assert!(matches!(
            tf32,
            Err(WgmmaError::MnMajor {
                operand: MatrixIdent::A,
                ..
            })
        ));
        let k = WgmmaASource::Shared(WgmmaMajor::K);
        let fp8 = WgmmaForm::new(elems(E4M3, E4M3, F32), shape(64, 32), k, WgmmaMajor::MN);
        assert!(matches!(
            fp8,
            Err(WgmmaError::MnMajor {
                operand: MatrixIdent::B,
                ..
            })
        ));
    }

    #[test]
    fn unsupported_shapes_are_refused() {
        let k = WgmmaASource::Shared(WgmmaMajor::K);
        // `n` past 256, `k` other than 32 bytes, and an integer `n` off its granularity of 16.
        for (elems, shape) in [
            (elems(F16, F16, F32), shape(264, 16)),
            (elems(F16, F16, F32), shape(64, 32)),
            (elems(S8, S8, S32), shape(40, 32)),
        ] {
            let form = WgmmaForm::new(elems, shape, k, WgmmaMajor::K);
            assert!(
                matches!(form, Err(WgmmaError::Unsupported { .. })),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn registers_must_fill_the_instruction() {
        let k = WgmmaASource::Shared(WgmmaMajor::K);
        let form = WgmmaForm::new(elems(F16, F16, F32), shape(64, 16), k, WgmmaMajor::K).unwrap();
        assert!(form.check_accumulator(32).is_ok());
        assert!(form.check_accumulator(16).is_err());
        assert!(form.check_a_registers(128).is_ok());
        assert!(form.check_a_registers(64).is_err());
    }

    #[test]
    fn descriptor_offsets_fit_their_fields() {
        let bits = descriptor_constant_bits(16, 1024, WgmmaSwizzle::B128).unwrap();
        assert_eq!(bits, 1 << 16 | 64 << 32 | 1 << 62);
        assert!(descriptor_constant_bits(8, 1024, WgmmaSwizzle::None).is_err());
        assert!(descriptor_constant_bits(16, 256 * 1024, WgmmaSwizzle::None).is_err());
    }
}

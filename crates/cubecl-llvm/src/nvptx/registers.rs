//! NVPTX registers: how a vector of elements is split into the registers an instruction or an
//! inline assembly takes, and joined back from what it returns.
//!
//! Matrix instructions take their operands as a list of 32-bit registers. A fragment held in a
//! vector is split one element per register, packed several to a register, or bitcast into opaque
//! 32-bit words, depending on what the instruction declares for its element type.

use crate::{prelude::*, shared::plane::bitcast};
use pliron_llvm::types::{StructLayout, StructType};

/// How a fragment is laid out in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fragment {
    pub(crate) regs: usize,
    /// Scalar elements per register.
    pub(crate) per_reg: usize,
}

impl Fragment {
    /// Scalar elements per lane, including duplicated input elements.
    pub(crate) fn elems(&self) -> usize {
        self.regs * self.per_reg
    }
}

pub(crate) fn extract_lane(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    i: usize,
) -> Value {
    let index = insert_i32_const(ctx, rw, i as i32);
    let op = llvm::ExtractElementOp::new(ctx, vector, index);
    insert(ctx, rw, &op)
}

pub(crate) fn insert_lane(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    value: Value,
    i: usize,
) -> Value {
    let index = insert_i32_const(ctx, rw, i as i32);
    let op = llvm::InsertElementOp::new(ctx, vector, value, index);
    insert(ctx, rw, &op)
}

pub(crate) fn poison(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    ty: TypeHandle,
) -> Value {
    let op = llvm::PoisonOp::new(ctx, ty);
    insert(ctx, rw, &op)
}

pub(crate) fn to_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    frag: Fragment,
    reg_ty: TypeHandle,
) -> Vec<Value> {
    (0..frag.regs)
        .map(|r| {
            if frag.per_reg == 1 {
                let value = extract_lane(ctx, rw, fragment, r);
                return bitcast(ctx, rw, value, reg_ty);
            }
            let lanes_ty = packed_ty(ctx, fragment.get_type(ctx), frag.per_reg);
            let mut reg = poison(ctx, rw, lanes_ty);
            for lane in 0..frag.per_reg {
                let element = extract_lane(ctx, rw, fragment, r * frag.per_reg + lane);
                reg = insert_lane(ctx, rw, reg, element, lane);
            }
            bitcast(ctx, rw, reg, reg_ty)
        })
        .collect()
}

pub(crate) fn from_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    regs: &[Value],
    frag: Fragment,
    frag_ty: TypeHandle,
) -> Value {
    let mut acc = poison(ctx, rw, frag_ty);
    for (r, &reg) in regs.iter().enumerate() {
        if frag.per_reg == 1 {
            let elem_ty = frag_ty
                .deref(ctx)
                .downcast_ref::<LlvmVectorType>()
                .unwrap()
                .elem_type();
            let reg = bitcast(ctx, rw, reg, elem_ty);
            acc = insert_lane(ctx, rw, acc, reg, r);
            continue;
        }
        let lanes_ty = packed_ty(ctx, frag_ty, frag.per_reg);
        let reg = bitcast(ctx, rw, reg, lanes_ty);
        for lane in 0..frag.per_reg {
            let element = extract_lane(ctx, rw, reg, lane);
            acc = insert_lane(ctx, rw, acc, element, r * frag.per_reg + lane);
        }
    }
    acc
}

/// The `per_reg` elements of `frag_ty` one register holds, as a vector. A register of another
/// type, such as the `i32` that carries two `bf16`, is a bitcast of it.
fn packed_ty(ctx: &Context, frag_ty: TypeHandle, per_reg: usize) -> TypeHandle {
    let elem = frag_ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .expect("a fragment is held in a vector")
        .elem_type();
    LlvmVectorType::get(ctx, elem, per_reg as u32, VectorTypeKind::Fixed).into()
}

pub(crate) fn call_returning_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    reg_tys: Vec<TypeHandle>,
    args: Vec<Value>,
) -> Vec<Value> {
    let count = reg_tys.len();
    let result_ty = registers_result_ty(ctx, reg_tys);
    let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let fn_ty = FuncType::get(ctx, result_ty, arg_tys, false);
    let call = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
    let result = insert(ctx, rw, &call);
    unpack_registers(ctx, rw, result, count)
}

/// The type an instruction returning `reg_tys` returns: the register itself when there is one,
/// a struct of them otherwise.
pub(crate) fn registers_result_ty(ctx: &mut Context, reg_tys: Vec<TypeHandle>) -> TypeHandle {
    if reg_tys.len() == 1 {
        reg_tys[0]
    } else {
        StructType::get_unnamed(ctx, (reg_tys, StructLayout::Unpacked)).into()
    }
}

/// The `count` registers in a value of [`registers_result_ty`].
pub(crate) fn unpack_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    result: Value,
    count: usize,
) -> Vec<Value> {
    if count == 1 {
        return vec![result];
    }
    (0..count)
        .map(|field| {
            let op = llvm::ExtractValueOp::new(ctx, result, vec![field as u32])
                .expect("a constant index into the returned registers");
            insert(ctx, rw, &op)
        })
        .collect()
}

/// Lanes of a vector of registers.
pub(crate) fn vector_lanes(ctx: &Context, ty: TypeHandle) -> usize {
    ty.deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .expect("registers are held in a vector")
        .num_elements() as usize
}

/// Bits of a vector of registers.
pub(crate) fn vector_bits(ctx: &Context, ty: TypeHandle) -> usize {
    let elem = ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .expect("registers are held in a vector")
        .elem_type();
    let elem = elem.deref(ctx);
    let bits = if let Some(int) = elem.downcast_ref::<IntegerType>() {
        int.width() as usize
    } else if elem.is::<FP64Type>() {
        64
    } else if elem.is::<FP32Type>() {
        32
    } else if elem.is::<FP16Type>() || elem.is::<BF16Type>() {
        16
    } else {
        unreachable!("registers hold integers or floats")
    };
    vector_lanes(ctx, ty) * bits
}

pub(crate) fn load_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::LoadOp::new(ctx, matrix, ty);
    insert(ctx, rw, &op)
}

pub(crate) fn store_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    value: Value,
) {
    let op = llvm::StoreOp::new(ctx, value, matrix);
    rw.insert_op(ctx, &op);
}

/// Register packing for manual matrix operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegisterForm {
    /// Several elements per register.
    Packed(usize),
    /// One element per register.
    Scalar,
    /// Elements packed into an opaque 32-bit register.
    Word,
}

pub(crate) fn word_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

impl RegisterForm {
    /// The registers that hold `vector`, in this form.
    pub(crate) fn split(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        vector: Value,
    ) -> Vec<Value> {
        let (elems, elem) = {
            let ty = vector.get_type(ctx);
            let ty = ty.deref(ctx);
            let vec = ty
                .downcast_ref::<LlvmVectorType>()
                .expect("a fragment is held in a vector");
            (vec.num_elements() as usize, vec.elem_type())
        };

        match self {
            RegisterForm::Scalar => (0..elems)
                .map(|i| extract_lane(ctx, rw, vector, i))
                .collect(),
            RegisterForm::Packed(per_reg) => {
                let reg_ty =
                    LlvmVectorType::get(ctx, elem, per_reg as u32, VectorTypeKind::Fixed).into();
                let frag = Fragment {
                    regs: elems / per_reg,
                    per_reg,
                };
                to_registers(ctx, rw, vector, frag, reg_ty)
            }
            RegisterForm::Word => {
                let word = word_ty(ctx);
                let words = vector_bits(ctx, vector.get_type(ctx)) / 32;
                let words_ty =
                    LlvmVectorType::get(ctx, word, words as u32, VectorTypeKind::Fixed).into();
                let as_words = bitcast(ctx, rw, vector, words_ty);
                (0..words)
                    .map(|i| extract_lane(ctx, rw, as_words, i))
                    .collect()
            }
        }
    }

    /// The vector of `vector_ty` that `regs`, in this form, hold: the inverse of [`Self::split`].
    pub(crate) fn join(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        regs: &[Value],
        vector_ty: TypeHandle,
    ) -> Value {
        let (elems, elem) = {
            let ty = vector_ty.deref(ctx);
            let vec = ty
                .downcast_ref::<LlvmVectorType>()
                .expect("a fragment is held in a vector");
            (vec.num_elements() as usize, vec.elem_type())
        };

        match self {
            RegisterForm::Scalar => {
                let mut acc = poison(ctx, rw, vector_ty);
                for (i, &reg) in regs.iter().enumerate() {
                    acc = insert_lane(ctx, rw, acc, reg, i);
                }
                acc
            }
            RegisterForm::Packed(per_reg) => {
                let frag = Fragment {
                    regs: elems / per_reg,
                    per_reg,
                };
                from_registers(ctx, rw, regs, frag, vector_ty)
            }
            RegisterForm::Word => {
                let word = word_ty(ctx);
                let words_ty =
                    LlvmVectorType::get(ctx, word, regs.len() as u32, VectorTypeKind::Fixed).into();
                let mut acc = poison(ctx, rw, words_ty);
                for (i, &reg) in regs.iter().enumerate() {
                    acc = insert_lane(ctx, rw, acc, reg, i);
                }
                let _ = elem;
                bitcast(ctx, rw, acc, vector_ty)
            }
        }
    }
}

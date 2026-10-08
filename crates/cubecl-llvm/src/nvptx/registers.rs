//! NVPTX registers: how a vector of elements is split into the registers an instruction or an
//! inline assembly takes, and joined back from what it returns.
//!
//! Matrix instructions take their operands as a list of 32-bit registers. A fragment held in a
//! vector is split one element per register, packed several to a register, or bitcast into opaque
//! 32-bit words, depending on what the instruction declares for its element type.

use crate::{
    prelude::*,
    shared::{
        matrix::{registers_array_ty, registers_as_vector, registers_value, vector_into_array},
        plane::bitcast,
    },
};
use pliron_llvm::types::{StructLayout, StructType};

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

pub(crate) fn call_returning_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    reg_tys: Vec<TypeHandle>,
    args: Vec<Value>,
) -> Vec<Value> {
    let count = reg_tys.len();
    let result_ty = registers_result_ty(ctx, reg_tys);
    let result = call_intrinsic(ctx, rw, name, result_ty, args);
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

impl RegisterForm {
    /// The type of one register holding elements of `elem` in this form.
    pub(crate) fn register_ty(self, ctx: &mut Context, elem: TypeHandle) -> TypeHandle {
        match self {
            RegisterForm::Scalar => elem,
            RegisterForm::Packed(per_reg) => {
                LlvmVectorType::get(ctx, elem, per_reg as u32, VectorTypeKind::Fixed).into()
            }
            RegisterForm::Word => i32_ty(ctx),
        }
    }

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
                let reg_ty = self.register_ty(ctx, elem);
                (0..elems / per_reg)
                    .map(|r| {
                        let mut reg = poison(ctx, rw, reg_ty);
                        for lane in 0..per_reg {
                            let element = extract_lane(ctx, rw, vector, r * per_reg + lane);
                            reg = insert_lane(ctx, rw, reg, element, lane);
                        }
                        reg
                    })
                    .collect()
            }
            RegisterForm::Word => {
                let word = i32_ty(ctx);
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
        match self {
            RegisterForm::Scalar => {
                let mut acc = poison(ctx, rw, vector_ty);
                for (i, &reg) in regs.iter().enumerate() {
                    acc = insert_lane(ctx, rw, acc, reg, i);
                }
                acc
            }
            RegisterForm::Packed(per_reg) => {
                let mut acc = poison(ctx, rw, vector_ty);
                for (r, &reg) in regs.iter().enumerate() {
                    for lane in 0..per_reg {
                        let element = extract_lane(ctx, rw, reg, lane);
                        acc = insert_lane(ctx, rw, acc, element, r * per_reg + lane);
                    }
                }
                acc
            }
            RegisterForm::Word => {
                let word = i32_ty(ctx);
                let words_ty =
                    LlvmVectorType::get(ctx, word, regs.len() as u32, VectorTypeKind::Fixed).into();
                let mut acc = poison(ctx, rw, words_ty);
                for (i, &reg) in regs.iter().enumerate() {
                    acc = insert_lane(ctx, rw, acc, reg, i);
                }
                bitcast(ctx, rw, acc, vector_ty)
            }
        }
    }
}

/// An array of registers a lowered operation reads and rewrites in place, split into the
/// registers an instruction or an inline assembly takes.
pub(crate) struct RegisterOperand {
    array: Value,
    vector_ty: TypeHandle,
    elem: TypeHandle,
    form: RegisterForm,
}

impl RegisterOperand {
    /// The registers of `array`, the converted value of a cube array or a pointer to one, taken
    /// in `form`.
    pub(crate) fn new(
        ctx: &Context,
        info: &OperandsInfo,
        array: Value,
        form: impl FnOnce(&Context, TypeHandle) -> RegisterForm,
    ) -> Self {
        let (vector_ty, elem) = registers_as_vector(ctx, info, array);
        Self {
            array,
            vector_ty,
            elem,
            form: form(ctx, elem),
        }
    }

    /// The cube type of an element.
    pub(crate) fn elem(&self) -> TypeHandle {
        self.elem
    }

    /// The elements the array holds.
    pub(crate) fn lanes(&self, ctx: &Context) -> usize {
        vector_lanes(ctx, self.vector_ty)
    }

    /// The bits the array holds.
    pub(crate) fn bits(&self, ctx: &Context) -> usize {
        vector_bits(ctx, self.vector_ty)
    }

    /// Reads the array, as registers.
    pub(crate) fn load(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Vec<Value> {
        let value = registers_value(ctx, rw, self.array, self.vector_ty);
        self.form.split(ctx, rw, value)
    }

    /// Writes `regs`, which [`Self::load`] read and an instruction rewrote, back to the array.
    pub(crate) fn store(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        info: &OperandsInfo,
        regs: &[Value],
    ) {
        let value = self.form.join(ctx, rw, regs, self.vector_ty);
        let array_ty = registers_array_ty(ctx, info, self.array);
        let value = vector_into_array(ctx, rw, value, array_ty);
        store_fragment(ctx, rw, self.array, value);
    }
}

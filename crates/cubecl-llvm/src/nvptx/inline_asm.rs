//! Inline PTX, for the instructions LLVM has no intrinsic for.

use crate::{
    nvptx::registers::{registers_result_ty, unpack_registers},
    prelude::*,
};
use pliron_llvm::llvm_attrs::{LlvmAttrValue, LlvmAttributesAttr};

/// An inline PTX statement and its operands, numbered as the template refers to them: the
/// registers it rewrites in place first (`$0..`, then the same registers again as the inputs tied
/// to them), then the plain inputs in the order they were added.
pub(crate) struct InlineAsm {
    tied: Vec<Value>,
    inputs: Vec<Value>,
    clobbers_memory: bool,
    convergent: bool,
}

impl InlineAsm {
    /// A statement that reads and writes `tied` in place. [`Self::emit`] returns what it wrote.
    /// They come first in the operand numbering, so they are fixed before any input is added.
    pub(crate) fn new(tied: Vec<Value>) -> Self {
        Self {
            tied,
            inputs: Vec::new(),
            clobbers_memory: false,
            convergent: false,
        }
    }

    /// Adds an input, read in the register class of its type, and returns the template operand
    /// that names it.
    pub(crate) fn input(&mut self, value: Value) -> String {
        let operand = format!("${}", 2 * self.tied.len() + self.inputs.len());
        self.inputs.push(value);
        operand
    }

    /// The statement reads or writes memory the compiler cannot see, so no memory access moves
    /// across it.
    pub(crate) fn clobbers_memory(mut self) -> Self {
        self.clobbers_memory = true;
        self
    }

    /// The statement is executed by a set of units together, so the compiler may not make it
    /// conditional on a value they disagree on.
    pub(crate) fn convergent(mut self) -> Self {
        self.convergent = true;
        self
    }

    /// The template operands naming the tied registers, braced as a PTX vector operand.
    pub(crate) fn tied_operands(&self) -> String {
        operand_list(0, self.tied.len())
    }

    /// Inserts the statement, which always has side effects, and returns the tied registers as
    /// it left them.
    pub(crate) fn emit(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        template: &str,
    ) -> Vec<Value> {
        let count = self.tied.len();
        let reg_tys: Vec<TypeHandle> = self.tied.iter().map(|reg| reg.get_type(ctx)).collect();

        let mut constraints: Vec<String> = self
            .tied
            .iter()
            .map(|&reg| format!("={}", RegisterClass::new(ctx, reg)))
            .collect();
        constraints.extend((0..count).map(|i| i.to_string()));
        constraints.extend(
            self.inputs
                .iter()
                .map(|&input| RegisterClass::new(ctx, input).to_string()),
        );
        if self.clobbers_memory {
            constraints.push("~{memory}".into());
        }

        let mut args = self.tied;
        args.extend(self.inputs);

        let result_ty = if count == 0 {
            VoidType::get(ctx).into()
        } else {
            registers_result_ty(ctx, reg_tys)
        };
        let has_side_effects = true;
        let asm = llvm::InlineAsmOp::new(
            ctx,
            result_ty,
            args,
            template,
            &constraints.join(","),
            has_side_effects,
        );
        if self.convergent {
            let mut attrs = LlvmAttributesAttr::new();
            attrs.set("convergent", LlvmAttrValue::Unit);
            asm.set_attr_llvm_inline_asm_attrs(ctx, attrs);
        }
        if count == 0 {
            rw.insert_op(ctx, &asm);
            return Vec::new();
        }
        let result = insert(ctx, rw, &asm);
        unpack_registers(ctx, rw, result, count)
    }
}

/// A run of `$i` operand references, braced as a PTX vector operand.
fn operand_list(first: usize, count: usize) -> String {
    let operands: Vec<String> = (first..first + count).map(|i| format!("${i}")).collect();
    format!("{{{}}}", operands.join(", "))
}

/// The PTX register class an operand is passed in, named by its LLVM constraint letter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegisterClass {
    B16,
    B32,
    B64,
    F32,
    F64,
}

impl RegisterClass {
    fn new(ctx: &Context, value: Value) -> Self {
        let ty = value.get_type(ctx);
        let ty = ty.deref(ctx);
        if ty.is::<FP32Type>() {
            return RegisterClass::F32;
        }
        if ty.is::<FP64Type>() {
            return RegisterClass::F64;
        }
        if ty.is::<FP16Type>() || ty.is::<BF16Type>() {
            return RegisterClass::B16;
        }
        match ty.downcast_ref::<IntegerType>().map(|int| int.width()) {
            Some(16) => RegisterClass::B16,
            Some(64) => RegisterClass::B64,
            Some(32) => RegisterClass::B32,
            _ => unreachable!("an inline PTX operand is a 16, 32 or 64-bit register"),
        }
    }
}

impl core::fmt::Display for RegisterClass {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let letter = match self {
            RegisterClass::B16 => "h",
            RegisterClass::B32 => "r",
            RegisterClass::B64 => "l",
            RegisterClass::F32 => "f",
            RegisterClass::F64 => "d",
        };
        f.write_str(letter)
    }
}

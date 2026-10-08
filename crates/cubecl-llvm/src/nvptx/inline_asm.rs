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
    inputs: Vec<(Value, &'static str)>,
    clobbers_memory: bool,
    convergent: bool,
}

impl InlineAsm {
    pub(crate) fn new() -> Self {
        Self {
            tied: Vec::new(),
            inputs: Vec::new(),
            clobbers_memory: false,
            convergent: false,
        }
    }

    /// Registers the statement reads and writes in place. [`Self::emit`] returns what it wrote.
    pub(crate) fn tied(mut self, regs: Vec<Value>) -> Self {
        self.tied = regs;
        self
    }

    /// Adds an input read through the register class `constraint`, and returns the template
    /// operand that names it.
    pub(crate) fn input(&mut self, value: Value, constraint: &'static str) -> String {
        let operand = format!("${}", 2 * self.tied.len() + self.inputs.len());
        self.inputs.push((value, constraint));
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
            .map(|&reg| format!("={}", register_class(ctx, reg)))
            .collect();
        constraints.extend((0..count).map(|i| i.to_string()));
        constraints.extend(self.inputs.iter().map(|(_, c)| c.to_string()));
        if self.clobbers_memory {
            constraints.push("~{memory}".into());
        }

        let mut args = self.tied;
        args.extend(self.inputs.into_iter().map(|(value, _)| value));

        let result_ty = if count == 0 {
            VoidType::get(ctx).into()
        } else {
            registers_result_ty(ctx, reg_tys)
        };
        let asm =
            llvm::InlineAsmOp::new(ctx, result_ty, args, template, &constraints.join(","), true);
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
pub(crate) fn operand_list(first: usize, count: usize) -> String {
    let operands: Vec<String> = (first..first + count).map(|i| format!("${i}")).collect();
    format!("{{{}}}", operands.join(", "))
}

/// The PTX register class of `reg`: `f` for `f32`, `l` for 64 bits, `r` for any other 32 bits.
fn register_class(ctx: &Context, reg: Value) -> &'static str {
    let ty = reg.get_type(ctx);
    let ty = ty.deref(ctx);
    if ty.is::<FP32Type>() {
        "f"
    } else if ty
        .downcast_ref::<IntegerType>()
        .is_some_and(|int| int.width() == 64)
    {
        "l"
    } else {
        "r"
    }
}

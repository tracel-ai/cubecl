use cubecl_core::{
    define_scalar, define_size,
    ir::{
        Arithmetic, Bitwise, ElemType, Instruction, IntKind, Operation, Operator, Scope, Type,
        UIntKind, UnaryOperands, Variable,
    },
    prelude::{assign, expand_erf, expand_hypot, expand_rhypot},
};
use cubecl_opt::{IrTransformer, TransformAction};

use crate::bitwise::{
    small_int_reverse, u16_u8_leading_zeros, u16_u8_trailing_zeros, u64_count_bits, u64_ffs,
    u64_leading_zeros, u64_reverse, u64_trailing_zeros,
};

define_scalar!(IntA);
define_size!(SizeA);

/// Expand erf
#[derive(Debug)]
pub(crate) struct ErfTransform;

impl IrTransformer for ErfTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                let scope = scope.child();
                expand_erf(&scope, op.input, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Expand hypot
#[derive(Debug)]
pub(crate) struct HypotTransform;

impl IrTransformer for HypotTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                let scope = scope.child();
                expand_hypot(&scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Expand hypot
#[derive(Debug)]
pub(crate) struct RhypotTransform;

impl IrTransformer for RhypotTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                let scope = scope.child();
                expand_rhypot(&scope, op.lhs, op.rhs, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Transform operations that only support 32 bits using polyfills
#[derive(Debug)]
pub(crate) struct BitwiseTransform {
    /// Allow base (non-extension) instructions with arbitrary bit widths. As far as I can tell,
    /// extension functions are still limited with maintenance9.
    arbitrary_bitwise: bool,
}

impl BitwiseTransform {
    pub(crate) fn new(arbitrary_bitwise: bool) -> Self {
        Self { arbitrary_bitwise }
    }
}

impl IrTransformer for BitwiseTransform {
    fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
        let op = match &inst.operation {
            Operation::Bitwise(op) => op,
            _ => return TransformAction::Ignore,
        };
        match op {
            Bitwise::CountOnes(op) if is_u64(op.input) && !self.arbitrary_bitwise => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_count_bits::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::CountOnes(op) if is_u16_u8(op.input) && !self.arbitrary_bitwise => {
                let u32 = Type::new(UIntKind::U32.into()).with_vector_size(op.input.vector_size());
                let tmp = scope.create_local(u32);
                let cast = Instruction::new(Operator::Cast(UnaryOperands { input: op.input }), tmp);
                let op =
                    Instruction::new(Bitwise::CountOnes(UnaryOperands { input: tmp }), inst.out());
                TransformAction::Replace(vec![cast, op])
            }
            Bitwise::ReverseBits(op)
                if op.input.storage_type().size() != 4 && !self.arbitrary_bitwise =>
            {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.ty.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let input = op.input;
                match op.input.storage_type().size() {
                    8 => {
                        let res = u64_reverse::expand::<IntA, SizeA>(&scope, input.into());
                        assign::expand_no_check(&scope, res, &mut inst.out().into());
                        TransformAction::Replace(into_instructions(scope))
                    }
                    width => {
                        let res = small_int_reverse::expand::<IntA, SizeA>(
                            &scope,
                            input.into(),
                            width as u32 * 8,
                        );
                        assign::expand_no_check(&scope, res, &mut inst.out().into());
                        TransformAction::Replace(into_instructions(scope))
                    }
                }
            }
            Bitwise::LeadingZeros(op) if is_u64(op.input) => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_leading_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::LeadingZeros(op) if is_u16_u8(op.input) => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u16_u8_leading_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::FindFirstSet(op) if is_u64(op.input) => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_ffs::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::FindFirstSet(op) if is_u16_u8(op.input) => {
                let u32 = Type::new(UIntKind::U32.into()).with_vector_size(op.input.vector_size());
                let tmp = scope.create_local(u32);
                let cast = Instruction::new(Operator::Cast(UnaryOperands { input: op.input }), tmp);
                let op = Instruction::new(
                    Bitwise::FindFirstSet(UnaryOperands { input: tmp }),
                    inst.out(),
                );
                TransformAction::Replace(vec![cast, op])
            }
            Bitwise::TrailingZeros(op) if is_u64(op.input) => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_trailing_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::TrailingZeros(op) if is_u16_u8(op.input) => {
                let scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u16_u8_trailing_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
                assign::expand_no_check(&scope, res, &mut inst.out().into());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

fn is_u64(var: Variable) -> bool {
    matches!(
        var.ty.elem_type(),
        ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64)
    )
}

fn is_u16_u8(var: Variable) -> bool {
    matches!(
        var.ty.elem_type(),
        ElemType::Int(IntKind::I16)
            | ElemType::UInt(UIntKind::U16)
            | ElemType::Int(IntKind::I8)
            | ElemType::UInt(UIntKind::U8)
    )
}

fn into_instructions(scope: Scope) -> Vec<Instruction> {
    scope.process([]).instructions
}

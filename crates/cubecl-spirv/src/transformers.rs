use cubecl_core::{
    ir::{
        Arithmetic, Bitwise, Elem, ExpandElement, Instruction, IntKind, Operation, Scope, UIntKind,
        Variable,
    },
    prelude::{IntExpand, expand_erf},
};
use cubecl_opt::{IrTransformer, TransformAction};

use crate::bitwise::{small_int_reverse, u64_count_bits, u64_ffs, u64_leading_zeros, u64_reverse};

/// Expand erf
#[derive(Debug)]
pub(crate) struct ErfTransform;

impl IrTransformer for ErfTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Erf(op)) => {
                let mut scope = scope.child();
                expand_erf(&mut scope, op.input, inst.out.unwrap());
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

/// Transform operations that only support 32 bits using polyfills
#[derive(Debug)]
pub(crate) struct BitwiseTransform;

impl IrTransformer for BitwiseTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        let op = match &inst.operation {
            Operation::Bitwise(op) => op,
            _ => return TransformAction::Ignore,
        };
        match op {
            Bitwise::CountOnes(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_elem::<IntExpand<0>>(op.input.elem());
                u64_count_bits::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                    ExpandElement::Plain(inst.out()).into(),
                );
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::ReverseBits(op) if op.input.elem().size() != 4 => {
                let mut scope = scope.child();
                scope.register_elem::<IntExpand<0>>(op.input.item.elem);
                let input = ExpandElement::Plain(op.input);
                let out = ExpandElement::Plain(inst.out());
                match op.input.elem().size() {
                    8 => {
                        u64_reverse::expand::<IntExpand<0>>(&mut scope, input.into(), out.into());
                        TransformAction::Replace(into_instructions(scope))
                    }
                    width => {
                        small_int_reverse::expand::<IntExpand<0>>(
                            &mut scope,
                            input.into(),
                            out.into(),
                            width as u32 * 8,
                        );
                        TransformAction::Replace(into_instructions(scope))
                    }
                }
            }
            Bitwise::LeadingZeros(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_elem::<IntExpand<0>>(op.input.elem());
                u64_leading_zeros::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                    ExpandElement::Plain(inst.out()).into(),
                );
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::FindFirstSet(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_elem::<IntExpand<0>>(op.input.elem());
                u64_ffs::expand::<IntExpand<0>>(
                    &mut scope,
                    ExpandElement::Plain(op.input).into(),
                    ExpandElement::Plain(inst.out()).into(),
                );
                TransformAction::Replace(into_instructions(scope))
            }
            _ => TransformAction::Ignore,
        }
    }
}

fn is_u64(var: Variable) -> bool {
    matches!(
        var.item.elem,
        Elem::Int(IntKind::I64) | Elem::UInt(UIntKind::U64)
    )
}

fn into_instructions(mut scope: Scope) -> Vec<Instruction> {
    scope.process([]).instructions
}

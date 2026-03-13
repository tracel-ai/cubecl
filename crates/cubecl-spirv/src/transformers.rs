use cubecl_core::{
    define_elem, define_size,
    ir::{
        Arithmetic, Bitwise, ElemType, Instruction, IntKind, ManagedVariable, Operation, Scope,
        UIntKind, Variable,
    },
    prelude::{assign, expand_erf, expand_hypot, expand_rhypot},
};
use cubecl_opt::{IrTransformer, TransformAction};

use crate::bitwise::{small_int_reverse, u64_count_bits, u64_ffs, u64_leading_zeros, u64_reverse};

define_elem!(ElemA);
define_elem!(IntA);
define_size!(SizeA);

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

/// Expand hypot
#[derive(Debug)]
pub(crate) struct HypotTransform;

impl IrTransformer for HypotTransform {
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Hypot(op)) => {
                let mut scope = scope.child();
                expand_hypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
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
    fn maybe_transform(&self, scope: &mut Scope, inst: &Instruction) -> TransformAction {
        match &inst.operation {
            Operation::Arithmetic(Arithmetic::Rhypot(op)) => {
                let mut scope = scope.child();
                expand_rhypot(&mut scope, op.lhs, op.rhs, inst.out.unwrap());
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
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_count_bits::expand::<IntA, SizeA>(
                    &mut scope,
                    ManagedVariable::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ManagedVariable::Plain(inst.out()).into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::ReverseBits(op) if op.input.storage_type().size() != 4 => {
                let mut scope = scope.child();
                scope.register_type::<IntA>(op.input.ty.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let input = ManagedVariable::Plain(op.input);
                match op.input.storage_type().size() {
                    8 => {
                        let res = u64_reverse::expand::<IntA, SizeA>(&mut scope, input.into());
                        assign::expand_no_check(
                            &mut scope,
                            res,
                            ManagedVariable::Plain(inst.out()).into(),
                        );
                        TransformAction::Replace(into_instructions(scope))
                    }
                    width => {
                        let res = small_int_reverse::expand::<IntA, SizeA>(
                            &mut scope,
                            input.into(),
                            width as u32 * 8,
                        );
                        assign::expand_no_check(
                            &mut scope,
                            res,
                            ManagedVariable::Plain(inst.out()).into(),
                        );
                        TransformAction::Replace(into_instructions(scope))
                    }
                }
            }
            Bitwise::LeadingZeros(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_leading_zeros::expand::<IntA, SizeA>(
                    &mut scope,
                    ManagedVariable::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ManagedVariable::Plain(inst.out()).into());
                TransformAction::Replace(into_instructions(scope))
            }
            Bitwise::FindFirstSet(op) if is_u64(op.input) => {
                let mut scope = scope.child();
                scope.register_type::<IntA>(op.input.storage_type());
                scope.register_size::<SizeA>(op.input.vector_size());
                let res = u64_ffs::expand::<IntA, SizeA>(
                    &mut scope,
                    ManagedVariable::Plain(op.input).into(),
                );
                assign::expand_no_check(&mut scope, res, ManagedVariable::Plain(inst.out()).into());
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

fn into_instructions(mut scope: Scope) -> Vec<Instruction> {
    scope.process([]).instructions
}

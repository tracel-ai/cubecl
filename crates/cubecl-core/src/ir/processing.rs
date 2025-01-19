use crate::prelude::AtomicOp;

use super::{
    Branch, CoopMma, Elem, Instruction, Metadata, Operation, Operator, UIntKind, Variable,
    VariableKind,
};

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub operations: Vec<Instruction>,
}

impl ScopeProcessing {
    /// Optimize the [variables](Variable) and [operations](Operation).
    ///
    /// ## Notes:
    ///
    /// This should be called once right after the creation of the type.
    /// If you built this type from the [scope process function](super::Scope::process), you don't have to
    /// call it again.
    pub fn optimize(self) -> Self {
        self.sanitize_constant_scalars()
    }

    /// Make sure constant scalars are of the correct type so compilers don't have to do conversion
    /// and handle edge cases such as indexing with a signed integer.
    fn sanitize_constant_scalars(mut self) -> Self {
        self.operations
            .iter_mut()
            .for_each(|inst| match &mut inst.operation {
                Operation::Copy(op) => {
                    sanitize_constant_scalar_ref_var(op, &inst.out.unwrap());
                }
                Operation::Operator(op) => match op {
                    Operator::Add(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Fma(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.a, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.b, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.c, &inst.out.unwrap());
                    }
                    Operator::Sub(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Mul(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Div(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::MulHi(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Abs(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Exp(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Log(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Log1p(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Cos(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Sin(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Tanh(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Powf(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Sqrt(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Round(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Floor(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Ceil(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Erf(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Recip(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Equal(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::NotEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::Lower(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::Clamp(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.min_value, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.max_value, &inst.out.unwrap());
                    }
                    Operator::Greater(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::LowerEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::GreaterEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::Modulo(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Slice(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(&mut op.start, Elem::UInt(UIntKind::U32));
                        sanitize_constant_scalar_ref_elem(&mut op.end, Elem::UInt(UIntKind::U32));
                    }
                    Operator::Index(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(&mut op.rhs, Elem::UInt(UIntKind::U32));
                    }
                    Operator::UncheckedIndex(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(&mut op.rhs, Elem::UInt(UIntKind::U32));
                    }
                    Operator::IndexAssign(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.lhs, Elem::UInt(UIntKind::U32));
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::UncheckedIndexAssign(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.lhs, Elem::UInt(UIntKind::U32));
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::And(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::Or(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Operator::Not(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.input, Elem::Bool);
                    }
                    Operator::Neg(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap())
                    }
                    Operator::Max(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Min(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::BitwiseAnd(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::BitwiseOr(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::BitwiseXor(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::CountOnes(_) => {
                        // Nothing to do
                    }
                    Operator::ReverseBits(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::ShiftLeft(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::ShiftRight(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Remainder(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::Bitcast(_) => {}
                    Operator::Magnitude(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Normalize(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Operator::Dot(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Operator::InitLine(_) => {
                        // TODO: Sanitize based on elem
                    }
                    Operator::CopyMemory(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.in_index,
                            Elem::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_elem(
                            &mut op.out_index,
                            Elem::UInt(UIntKind::U32),
                        );
                    }
                    Operator::CopyMemoryBulk(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.in_index,
                            Elem::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_elem(
                            &mut op.out_index,
                            Elem::UInt(UIntKind::U32),
                        );
                    }
                    Operator::Select(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, Elem::Bool);
                        sanitize_constant_scalar_ref_var(&mut op.then, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.or_else, &inst.out.unwrap());
                    }
                    Operator::Cast(_) => {}
                },
                Operation::Atomic(op) => match op {
                    AtomicOp::Load(_) => {}
                    AtomicOp::Store(_) => {}
                    AtomicOp::Swap(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::CompareAndSwap(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.cmp, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.val, &inst.out.unwrap());
                    }
                    AtomicOp::Add(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::Sub(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::Max(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::Min(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::And(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::Or(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    AtomicOp::Xor(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                },
                Operation::Metadata(op) => match op {
                    Metadata::Stride { dim, .. } => {
                        sanitize_constant_scalar_ref_elem(dim, Elem::UInt(UIntKind::U32));
                    }
                    Metadata::Shape { dim, .. } => {
                        sanitize_constant_scalar_ref_elem(dim, Elem::UInt(UIntKind::U32));
                    }
                    Metadata::Length { .. }
                    | Metadata::BufferLength { .. }
                    | Metadata::Rank { .. } => {
                        // Nothing to do
                    }
                },
                Operation::Branch(op) => match op {
                    Branch::If(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, Elem::Bool);
                    }
                    Branch::IfElse(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, Elem::Bool);
                    }
                    Branch::RangeLoop(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.end, &op.start);
                        sanitize_constant_scalar_ref_var(&mut op.i, &op.start);
                        if let Some(step) = &mut op.step {
                            sanitize_constant_scalar_ref_elem(step, Elem::UInt(UIntKind::U32));
                        }
                    }
                    _ => {
                        // Nothing to do.
                    }
                },
                Operation::Synchronization(_) => {
                    // Nothing to do.
                }
                Operation::Plane(_) => {
                    // Nothing to do since no constant is possible.
                }
                Operation::CoopMma(op) => match op {
                    CoopMma::Fill { value } => {
                        sanitize_constant_scalar_ref_var(value, &inst.out.unwrap());
                    }
                    CoopMma::Load { value, stride, .. } => {
                        sanitize_constant_scalar_ref_var(value, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(stride, Elem::UInt(UIntKind::U32));
                    }
                    CoopMma::Execute { .. } => {
                        // Nothing to do.
                    }
                    CoopMma::Store { stride, .. } => {
                        sanitize_constant_scalar_ref_elem(stride, Elem::UInt(UIntKind::U32));
                    }
                    CoopMma::Cast { .. } => {
                        // Nothing to do.
                    }
                },
                Operation::NonSemantic(_) => {
                    // Nothing to do.
                }
                Operation::Pipeline(_) => {
                    // Nothing to do
                }
            });
        self
    }
}

fn sanitize_constant_scalar_ref_var(var: &mut Variable, reference: &Variable) {
    let elem = reference.item.elem();
    sanitize_constant_scalar_ref_elem(var, elem);
}

fn sanitize_constant_scalar_ref_elem(var: &mut Variable, elem: Elem) {
    if let VariableKind::ConstantScalar(scalar) = var.kind {
        if scalar.elem() != elem {
            *var = match scalar {
                super::ConstantScalarValue::Int(val, _) => elem.constant_from_i64(val),
                super::ConstantScalarValue::Float(val, _) => elem.constant_from_f64(val),
                super::ConstantScalarValue::UInt(val, _) => elem.constant_from_u64(val),
                super::ConstantScalarValue::Bool(val) => elem.constant_from_bool(val),
            };
        }
    }
}

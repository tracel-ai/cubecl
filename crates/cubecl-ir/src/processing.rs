use core::fmt::Display;

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::{Allocator, AtomicOp, Bitwise, Comparison, Operator};

use super::{
    Arithmetic, Branch, CoopMma, ElemType, Instruction, Metadata, Operation, UIntKind, Variable,
    VariableKind,
};

pub trait Processor: core::fmt::Debug {
    fn transform(&self, processing: ScopeProcessing, allocator: Allocator) -> ScopeProcessing;
}

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub instructions: Vec<Instruction>,
}

impl Display for ScopeProcessing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{{")?;
        for instruction in self.instructions.iter() {
            let instruction_str = instruction.to_string();
            if !instruction_str.is_empty() {
                writeln!(f, "    {instruction_str}")?;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
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
        self.instructions
            .iter_mut()
            .for_each(|inst| match &mut inst.operation {
                Operation::Copy(op) => {
                    sanitize_constant_scalar_ref_var(op, &inst.out.unwrap());
                }
                Operation::Arithmetic(op) => match op {
                    Arithmetic::Add(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::SaturatingAdd(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Fma(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.a, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.b, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.c, &inst.out.unwrap());
                    }
                    Arithmetic::Sub(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::SaturatingSub(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Mul(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Div(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::MulHi(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Abs(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Exp(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Log(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Log1p(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Cos(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Sin(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Tanh(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Powf(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Powi(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                    }
                    Arithmetic::Sqrt(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Round(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Floor(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Ceil(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Erf(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Recip(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Clamp(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.min_value, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.max_value, &inst.out.unwrap());
                    }
                    Arithmetic::Modulo(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Neg(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap())
                    }
                    Arithmetic::Max(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Min(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Remainder(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Arithmetic::Magnitude(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Normalize(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Arithmetic::Dot(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                },
                Operation::Comparison(op) => match op {
                    Comparison::Greater(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::LowerEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::GreaterEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::Equal(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::NotEqual(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::Lower(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &op.rhs);
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &op.lhs);
                    }
                    Comparison::IsNan(_op) | Comparison::IsInf(_op) => {
                        // Nothing to do
                    }
                },
                Operation::Bitwise(op) => match op {
                    Bitwise::BitwiseAnd(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Bitwise::BitwiseOr(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Bitwise::BitwiseXor(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Bitwise::CountOnes(_) | Bitwise::LeadingZeros(_) | Bitwise::FindFirstSet(_) => {
                        // Nothing to do
                    }
                    Bitwise::ReverseBits(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                    Bitwise::ShiftLeft(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Bitwise::ShiftRight(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.lhs, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.rhs, &inst.out.unwrap());
                    }
                    Bitwise::BitwiseNot(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                    }
                },
                Operation::Operator(op) => match op {
                    Operator::Index(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.list, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.index,
                            ElemType::UInt(UIntKind::U32),
                        );
                    }
                    Operator::UncheckedIndex(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.list, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.index,
                            ElemType::UInt(UIntKind::U32),
                        );
                    }
                    Operator::IndexAssign(op) => {
                        sanitize_constant_scalar_ref_elem(
                            &mut op.index,
                            ElemType::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_var(&mut op.value, &inst.out.unwrap());
                    }
                    Operator::UncheckedIndexAssign(op) => {
                        sanitize_constant_scalar_ref_elem(
                            &mut op.index,
                            ElemType::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_var(&mut op.value, &inst.out.unwrap());
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
                        sanitize_constant_scalar_ref_elem(&mut op.input, ElemType::Bool);
                    }
                    Operator::InitLine(_) => {
                        // TODO: Sanitize based on elem
                    }
                    Operator::CopyMemory(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.in_index,
                            ElemType::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_elem(
                            &mut op.out_index,
                            ElemType::UInt(UIntKind::U32),
                        );
                    }
                    Operator::CopyMemoryBulk(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.input, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_elem(
                            &mut op.in_index,
                            ElemType::UInt(UIntKind::U32),
                        );
                        sanitize_constant_scalar_ref_elem(
                            &mut op.out_index,
                            ElemType::UInt(UIntKind::U32),
                        );
                    }
                    Operator::Select(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, ElemType::Bool);
                        sanitize_constant_scalar_ref_var(&mut op.then, &inst.out.unwrap());
                        sanitize_constant_scalar_ref_var(&mut op.or_else, &inst.out.unwrap());
                    }
                    Operator::Cast(_) => {}
                    Operator::Reinterpret(_) => {}
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
                        sanitize_constant_scalar_ref_elem(dim, ElemType::UInt(UIntKind::U32));
                    }
                    Metadata::Shape { dim, .. } => {
                        sanitize_constant_scalar_ref_elem(dim, ElemType::UInt(UIntKind::U32));
                    }
                    Metadata::Length { .. }
                    | Metadata::BufferLength { .. }
                    | Metadata::Rank { .. } => {
                        // Nothing to do
                    }
                },
                Operation::Branch(op) => match op {
                    Branch::If(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, ElemType::Bool);
                    }
                    Branch::IfElse(op) => {
                        sanitize_constant_scalar_ref_elem(&mut op.cond, ElemType::Bool);
                    }
                    Branch::RangeLoop(op) => {
                        sanitize_constant_scalar_ref_var(&mut op.end, &op.start);
                        sanitize_constant_scalar_ref_var(&mut op.i, &op.start);
                        if let Some(step) = &mut op.step {
                            sanitize_constant_scalar_ref_elem(step, ElemType::UInt(UIntKind::U32));
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
                        sanitize_constant_scalar_ref_elem(stride, ElemType::UInt(UIntKind::U32));
                    }
                    CoopMma::Execute { .. }
                    | CoopMma::ExecuteManual { .. }
                    | CoopMma::ExecuteScaled { .. } => {
                        // Nothing to do.
                    }
                    CoopMma::Store { stride, .. } => {
                        sanitize_constant_scalar_ref_elem(stride, ElemType::UInt(UIntKind::U32));
                    }
                    CoopMma::Cast { .. } => {
                        // Nothing to do.
                    }
                    CoopMma::RowIndex { lane_id, i, .. } => {
                        sanitize_constant_scalar_ref_elem(lane_id, ElemType::UInt(UIntKind::U32));
                        sanitize_constant_scalar_ref_elem(i, ElemType::UInt(UIntKind::U32));
                    }
                    CoopMma::ColIndex { lane_id, i, .. } => {
                        sanitize_constant_scalar_ref_elem(lane_id, ElemType::UInt(UIntKind::U32));
                        sanitize_constant_scalar_ref_elem(i, ElemType::UInt(UIntKind::U32));
                    }
                },
                Operation::NonSemantic(_) => {
                    // Nothing to do.
                }
                Operation::Barrier(_) => {
                    // Nothing to do
                }
                Operation::Tma(_) => {
                    // Nothing to do
                }
                Operation::Free(_) => {
                    // Nothing to do
                }
            });
        self
    }
}

fn sanitize_constant_scalar_ref_var(var: &mut Variable, reference: &Variable) {
    let elem = reference.ty.elem_type();
    sanitize_constant_scalar_ref_elem(var, elem);
}

fn sanitize_constant_scalar_ref_elem(var: &mut Variable, elem: ElemType) {
    if let VariableKind::ConstantScalar(scalar) = var.kind
        && scalar.elem_type() != elem
    {
        *var = match scalar {
            super::ConstantScalarValue::Int(val, _) => elem.constant_from_i64(val),
            super::ConstantScalarValue::Float(val, _) => elem.constant_from_f64(val),
            super::ConstantScalarValue::UInt(val, _) => elem.constant_from_u64(val),
            super::ConstantScalarValue::Bool(val) => elem.constant_from_bool(val),
        };
    }
}

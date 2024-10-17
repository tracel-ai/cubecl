use super::{Branch, CoopMma, Elem, Metadata, Operation, Operator, Variable};

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub operations: Vec<Operation>,
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
        self.operations.iter_mut().for_each(|op| match op {
            Operation::Operator(op) => match op {
                Operator::Add(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Fma(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.a, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.b, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.c, &op.out);
                }
                Operator::Sub(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Mul(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Div(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Abs(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Exp(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Log(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Log1p(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Cos(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Sin(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Tanh(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Powf(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Sqrt(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Round(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Floor(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Ceil(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Erf(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Recip(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
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
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.min_value, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.max_value, &op.out);
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
                Operator::Assign(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Modulo(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Slice(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                    sanitize_constant_scalar_ref_elem(&mut op.start, Elem::UInt);
                    sanitize_constant_scalar_ref_elem(&mut op.end, Elem::UInt);
                }
                Operator::Index(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_elem(&mut op.rhs, Elem::UInt);
                }
                Operator::UncheckedIndex(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_elem(&mut op.rhs, Elem::UInt);
                }
                Operator::IndexAssign(op) => {
                    sanitize_constant_scalar_ref_elem(&mut op.lhs, Elem::UInt);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::UncheckedIndexAssign(op) => {
                    sanitize_constant_scalar_ref_elem(&mut op.lhs, Elem::UInt);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
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
                Operator::Neg(op) => sanitize_constant_scalar_ref_var(&mut op.input, &op.out),
                Operator::Max(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Min(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::BitwiseAnd(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::BitwiseOr(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::BitwiseXor(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::ShiftLeft(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::ShiftRight(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Remainder(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Bitcast(_) => {}
                Operator::AtomicLoad(_) => {}
                Operator::AtomicStore(_) => {}
                Operator::AtomicSwap(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicCompareAndSwap(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.cmp, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.val, &op.out);
                }
                Operator::AtomicAdd(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicSub(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicMax(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicMin(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicAnd(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicOr(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicXor(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::Magnitude(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Normalize(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                }
                Operator::Dot(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.lhs, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::InitLine(_) => {
                    // TODO: Sanitize based on elem
                }
                Operator::Copy(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                    sanitize_constant_scalar_ref_elem(&mut op.in_index, Elem::UInt);
                    sanitize_constant_scalar_ref_elem(&mut op.out_index, Elem::UInt);
                }
                Operator::CopyBulk(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.input, &op.out);
                    sanitize_constant_scalar_ref_elem(&mut op.in_index, Elem::UInt);
                    sanitize_constant_scalar_ref_elem(&mut op.out_index, Elem::UInt);
                }
            },
            Operation::Metadata(op) => match op {
                Metadata::Stride { dim, .. } => {
                    sanitize_constant_scalar_ref_elem(dim, Elem::UInt);
                }
                Metadata::Shape { dim, .. } => {
                    sanitize_constant_scalar_ref_elem(dim, Elem::UInt);
                }
                Metadata::Length { .. } => {
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
                        sanitize_constant_scalar_ref_elem(step, Elem::UInt);
                    }
                }
                _ => {
                    // Nothing to do.
                }
            },
            Operation::Synchronization(_) => {
                // Nothing to do.
            }
            Operation::Subcube(_) => {
                // Nothing to do since no constant is possible.
            }
            Operation::CoopMma(op) => match op {
                CoopMma::Fill { mat, value } => {
                    sanitize_constant_scalar_ref_var(value, mat);
                }
                CoopMma::Load {
                    mat, value, stride, ..
                } => {
                    sanitize_constant_scalar_ref_var(value, mat);
                    sanitize_constant_scalar_ref_elem(stride, Elem::UInt);
                }
                CoopMma::Execute { .. } => {
                    // Nothing to do.
                }
                CoopMma::Store { stride, .. } => {
                    sanitize_constant_scalar_ref_elem(stride, Elem::UInt);
                }
            },
        });
        self
    }
}

fn sanitize_constant_scalar_ref_var(var: &mut Variable, reference: &Variable) {
    let elem = reference.item().elem();
    sanitize_constant_scalar_ref_elem(var, elem);
}

fn sanitize_constant_scalar_ref_elem(var: &mut Variable, elem: Elem) {
    if let Variable::ConstantScalar(scalar) = var {
        if scalar.elem() != elem {
            *var = match scalar {
                super::ConstantScalarValue::Int(val, _) => elem.constant_from_i64(*val),
                super::ConstantScalarValue::Float(val, _) => elem.constant_from_f64(*val),
                super::ConstantScalarValue::UInt(val) => elem.constant_from_u64(*val),
                super::ConstantScalarValue::Bool(val) => elem.constant_from_bool(*val),
            };
        }
    }
}

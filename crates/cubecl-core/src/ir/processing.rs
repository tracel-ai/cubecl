use super::{Branch, CoopMma, Elem, Metadata, Operation, Operator, Procedure, Variable};
use crate::ir::ReadGlobalWithLayout;

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
            .merge_read_global_with_layout()
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
                Operator::AtomicLoad(_) => {}
                Operator::AtomicStore(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
                }
                Operator::AtomicCompareAndSwap(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.cmp, &op.out);
                    sanitize_constant_scalar_ref_var(&mut op.val, &op.out);
                }
                Operator::Bitcast(_) => {}
                Operator::AtomicAdd(op) => {
                    sanitize_constant_scalar_ref_var(&mut op.rhs, &op.out);
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
                    sanitize_constant_scalar_ref_elem(&mut op.start, Elem::UInt);
                    sanitize_constant_scalar_ref_elem(&mut op.end, Elem::UInt);
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
                CoopMma::Load { mat, value, stride } => {
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
            Operation::Procedure(_) => {
                // Nothing to do since they are re-processed.
            }
        });
        self
    }

    /// Merge all compatible [read global with layout procedures](ReadGlobalWithLayout).
    fn merge_read_global_with_layout(mut self) -> Self {
        #[derive(Default)]
        struct Optimization {
            merged_procs: Vec<MergedProc>,
        }

        #[derive(new)]
        struct MergedProc {
            proc: ReadGlobalWithLayout,
            positions: Vec<usize>,
        }

        impl Optimization {
            fn new(existing_operations: &[Operation]) -> Self {
                let mut optim = Self::default();

                existing_operations
                    .iter()
                    .enumerate()
                    .for_each(|(position, operation)| {
                        if let Operation::Procedure(Procedure::ReadGlobalWithLayout(proc)) =
                            operation
                        {
                            optim.register_one(proc, position);
                        }
                    });

                optim
            }

            fn register_one(&mut self, proc: &ReadGlobalWithLayout, position: usize) {
                for merged_proc in self.merged_procs.iter_mut() {
                    if let Some(merged) = merged_proc.proc.try_merge(proc) {
                        merged_proc.proc = merged;
                        merged_proc.positions.push(position);
                        return;
                    }
                }

                self.merged_procs
                    .push(MergedProc::new(proc.clone(), vec![position]));
            }

            fn apply(self, existing_operations: Vec<Operation>) -> Vec<Operation> {
                if self.merged_procs.is_empty() {
                    return existing_operations;
                }

                let mut operations = Vec::with_capacity(existing_operations.len());

                for (position, operation) in existing_operations.into_iter().enumerate() {
                    let mut is_merged_op = false;

                    for merged_proc in self.merged_procs.iter() {
                        if merged_proc.positions[0] == position {
                            operations.push(Operation::Procedure(Procedure::ReadGlobalWithLayout(
                                merged_proc.proc.clone(),
                            )));
                            is_merged_op = true;
                        }

                        if merged_proc.positions.contains(&position) {
                            is_merged_op = true;
                        }
                    }

                    if !is_merged_op {
                        operations.push(operation);
                    }
                }

                operations
            }
        }

        let optimization = Optimization::new(&self.operations);
        self.operations = optimization.apply(self.operations);
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

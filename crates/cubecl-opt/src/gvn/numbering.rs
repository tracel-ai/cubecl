use std::{
    collections::{HashSet, LinkedList},
    mem::swap,
};

use cubecl_core::ir::{
    Branch, ConstantScalarValue, Metadata, Operation, Operator, Subcube, Variable,
};

use super::{Builtin, Expression, Instruction, Local, OpId, Value, ValueTable};

impl ValueTable {
    pub fn maybe_insert_op(
        &mut self,
        op: &Operation,
        exp_gen: &mut LinkedList<(u32, Expression)>,
        added_exps: &mut HashSet<u32>,
    ) -> Result<(u32, Option<Value>, Expression), Option<Value>> {
        let expr = self.create_expr(op);
        if let Err(Some(val)) = expr {
            let num = self.lookup_or_add_value(val);
            exp_gen.push_back((num, Expression::Volatile(val)));
        }
        let (expr, val) = expr?;
        let num = self.lookup_or_add_expr(expr.clone(), val);
        if !added_exps.contains(&num) {
            exp_gen.push_back((num, expr.clone()));
            added_exps.insert(num);
        }
        Ok((num, val, expr))
    }

    pub fn value_of_expr(&mut self, expr: Expression) -> (u32, bool) {
        if let Some(existing) = self.expression_numbers.get(&expr) {
            (*existing, false)
        } else {
            let num = self.next_value_num;
            self.expression_numbers.insert(expr.clone(), num);
            self.expressions.insert(self.next_expr_num, expr);
            self.next_value_num += 1;
            self.next_expr_num += 1;
            (num, true)
        }
    }

    pub fn lookup_or_add_expr(&mut self, expr: Expression, value: Option<Value>) -> u32 {
        let num = self.value_of_expr(expr).0;
        if let Some(value) = value {
            self.value_numbers.insert(value, num);
            self.next_value_num += 1;
        }
        num
    }

    pub fn lookup_or_add_value(&mut self, value: Value) -> u32 {
        if let Some(existing) = self.value_numbers.get(&value) {
            *existing
        } else {
            let num = self.next_value_num;
            self.value_numbers.insert(value, num);
            self.next_value_num += 1;
            num
        }
    }

    pub fn lookup_or_add_var(&mut self, value: &Variable) -> Result<u32, Option<Value>> {
        let value = value_of_var(value).ok_or(None)?;
        if let Some(existing) = self.value_numbers.get(&value) {
            Ok(*existing)
        } else {
            let num = self.next_value_num;
            self.value_numbers.insert(value, num);
            self.next_value_num += 1;
            Ok(num)
        }
    }

    fn create_expr(
        &mut self,
        op: &Operation,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        match op {
            Operation::Operator(operator) => self.create_expr_op(operator),
            Operation::Metadata(metadata) => self.create_expr_meta(metadata),
            Operation::Subcube(op) => {
                let val = match op {
                    Subcube::Elect(op) => value_of_var(&op.out),
                    Subcube::Broadcast(op) => value_of_var(&op.out),
                    Subcube::All(op)
                    | Subcube::Any(op)
                    | Subcube::Sum(op)
                    | Subcube::Prod(op)
                    | Subcube::Min(op)
                    | Subcube::Max(op) => value_of_var(&op.out),
                };
                Err(val)
            }
            Operation::Branch(Branch::Select(op)) => {
                let cond = self.lookup_or_add_var(&op.cond)?;
                let then = self.lookup_or_add_var(&op.then)?;
                let or_else = self.lookup_or_add_var(&op.or_else)?;
                let expr = Instruction::new(OpId::Select, &[cond, then, or_else]);
                Ok((expr.into(), value_of_var(&op.out)))
            }
            Operation::Branch(_) | Operation::Synchronization(_) | Operation::CoopMma(_) => {
                Err(None)
            }
        }
    }

    fn create_expr_op(
        &mut self,
        operator: &Operator,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        let (expr, val) = match operator {
            Operator::Assign(op) => {
                let out = value_of_var(&op.out);
                let num = self.lookup_or_add_var(&op.input)?;
                (Expression::Copy(num), out)
            }

            // Commutative binop
            Operator::Add(op)
            | Operator::Mul(op)
            | Operator::And(op)
            | Operator::Or(op)
            | Operator::Equal(op)
            | Operator::NotEqual(op)
            | Operator::BitwiseAnd(op)
            | Operator::BitwiseOr(op)
            | Operator::BitwiseXor(op)
            | Operator::Max(op)
            | Operator::Min(op)
            | Operator::Dot(op) => {
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let id = id_of_op(operator);
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                }
                let expr = Instruction::commutative(id, &[lhs, rhs]);
                (expr.into(), out)
            }

            // Non-commutative binops
            Operator::Sub(op)
            | Operator::Div(op)
            | Operator::Powf(op)
            | Operator::Modulo(op)
            | Operator::Remainder(op)
            | Operator::Index(op)
            | Operator::UncheckedIndex(op)
            | Operator::ShiftLeft(op)
            | Operator::ShiftRight(op) => {
                let lhs = self.lookup_or_add_var(&op.lhs)?;
                let rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let id = id_of_op(operator);
                let expr = Instruction::new(id, &[lhs, rhs]);
                (expr.into(), out)
            }

            // Compare ops
            Operator::Lower(op)
            | Operator::Greater(op)
            | Operator::LowerEqual(op)
            | Operator::GreaterEqual(op) => {
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let mut op = id_of_op(operator);
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                    op = cmp_inverse(&op);
                }
                let expr = Instruction::new(op, &[lhs, rhs]);
                (expr.into(), out)
            }

            // Unary ops
            Operator::Abs(op)
            | Operator::Exp(op)
            | Operator::Log(op)
            | Operator::Log1p(op)
            | Operator::Cos(op)
            | Operator::Sin(op)
            | Operator::Tanh(op)
            | Operator::Sqrt(op)
            | Operator::Round(op)
            | Operator::Floor(op)
            | Operator::Ceil(op)
            | Operator::Erf(op)
            | Operator::Recip(op)
            | Operator::Not(op)
            | Operator::Neg(op)
            | Operator::Bitcast(op)
            | Operator::Magnitude(op)
            | Operator::Normalize(op) => {
                let input = self.lookup_or_add_var(&op.input)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &[input]);
                (expr.into(), out)
            }

            Operator::Fma(op) => {
                let mut a = self.lookup_or_add_var(&op.a)?;
                let mut b = self.lookup_or_add_var(&op.b)?;
                let c = self.lookup_or_add_var(&op.c)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                if a > b {
                    swap(&mut a, &mut b);
                }
                let expr = Instruction::new(op, &[a, b, c]);
                (expr.into(), out)
            }
            Operator::Clamp(op) => {
                let val = self.lookup_or_add_var(&op.input)?;
                let min = self.lookup_or_add_var(&op.min_value)?;
                let max = self.lookup_or_add_var(&op.max_value)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &[val, min, max]);
                (expr.into(), out)
            }
            Operator::InitLine(op) => {
                let operands = op.inputs.iter().map(|it| self.lookup_or_add_var(it));
                let operands = operands.collect::<Result<Vec<_>, _>>()?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &operands);
                (expr.into(), out)
            }

            Operator::AtomicSwap(op)
            | Operator::AtomicAdd(op)
            | Operator::AtomicSub(op)
            | Operator::AtomicMax(op)
            | Operator::AtomicMin(op)
            | Operator::AtomicAnd(op)
            | Operator::AtomicOr(op)
            | Operator::AtomicXor(op) => Err(value_of_var(&op.out))?,
            Operator::AtomicLoad(op) => Err(value_of_var(&op.out))?,
            Operator::AtomicCompareAndSwap(op) => Err(value_of_var(&op.out))?,
            Operator::AtomicStore(_) => Err(None)?,
            Operator::IndexAssign(_)
            | Operator::UncheckedIndexAssign(_)
            | Operator::Slice(_)
            | Operator::CopyBulk(_)
            | Operator::Copy(_) => Err(None)?,
        };
        Ok((expr, val))
    }

    fn create_expr_meta(
        &mut self,
        meta: &Metadata,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        let op = id_of_meta(meta);
        let (expr, val) = match meta {
            Metadata::Stride { dim, var, out } | Metadata::Shape { dim, var, out } => {
                let var = self.lookup_or_add_var(var)?;
                let dim = self.lookup_or_add_var(dim)?;
                let out = value_of_var(out);
                let expr = Instruction::new(op, &[var, dim]);
                (expr, out)
            }
            Metadata::Length { var, out } => {
                let out = value_of_var(out);
                let var = match var {
                    Variable::GlobalInputArray { .. }
                    | Variable::GlobalOutputArray { .. }
                    | Variable::Slice { .. }
                    | Variable::GlobalScalar { .. } => self.lookup_or_add_var(var)?,
                    Variable::ConstantArray { length, .. }
                    | Variable::SharedMemory { length, .. }
                    | Variable::LocalArray { length, .. } => {
                        let constant =
                            Variable::ConstantScalar(ConstantScalarValue::UInt(*length as u64));
                        let num = self.lookup_or_add_var(&constant)?;
                        let expr = Expression::Copy(num);
                        return Ok((expr, out));
                    }
                    _ => unreachable!("Length only available on array"),
                };
                let expr = Instruction::new(op, &[var]);
                (expr, out)
            }
        };
        Ok((expr.into(), val))
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let val = match var {
        Variable::GlobalInputArray { id, .. } => Value::Input(*id),
        Variable::GlobalScalar { id, elem } => Value::Scalar(*id, *elem),
        Variable::GlobalOutputArray { id, .. } => Value::Output(*id),
        Variable::Versioned {
            id, depth, version, ..
        } => Value::Local(Local(*id, *depth, *version)),
        Variable::LocalBinding { id, depth, .. } => Value::Local(Local(*id, *depth, 0)),
        Variable::ConstantScalar(val) => Value::Constant((*val).into()),
        Variable::ConstantArray { id, .. } => Value::ConstArray(*id),
        Variable::Local { .. }
        | Variable::SharedMemory { .. }
        | Variable::LocalArray { .. }
        | Variable::Matrix { .. } => None?,
        Variable::Slice { id, depth, .. } => Value::Slice(*id, *depth),
        Variable::Rank => Value::Builtin(Builtin::Rank),
        Variable::UnitPos => Value::Builtin(Builtin::UnitPos),
        Variable::UnitPosX => Value::Builtin(Builtin::UnitPosX),
        Variable::UnitPosY => Value::Builtin(Builtin::UnitPosY),
        Variable::UnitPosZ => Value::Builtin(Builtin::UnitPosZ),
        Variable::CubePos => Value::Builtin(Builtin::CubePos),
        Variable::CubePosX => Value::Builtin(Builtin::CubePosX),
        Variable::CubePosY => Value::Builtin(Builtin::CubePosY),
        Variable::CubePosZ => Value::Builtin(Builtin::CubePosZ),
        Variable::CubeDim => Value::Builtin(Builtin::CubeDim),
        Variable::CubeDimX => Value::Builtin(Builtin::CubeDimX),
        Variable::CubeDimY => Value::Builtin(Builtin::CubeDimY),
        Variable::CubeDimZ => Value::Builtin(Builtin::CubeDimZ),
        Variable::CubeCount => Value::Builtin(Builtin::CubeCount),
        Variable::CubeCountX => Value::Builtin(Builtin::CubeCountX),
        Variable::CubeCountY => Value::Builtin(Builtin::CubeCountY),
        Variable::CubeCountZ => Value::Builtin(Builtin::CubeCountZ),
        Variable::SubcubeDim => Value::Builtin(Builtin::SubcubeDim),
        Variable::AbsolutePos => Value::Builtin(Builtin::AbsolutePos),
        Variable::AbsolutePosX => Value::Builtin(Builtin::AbsolutePosX),
        Variable::AbsolutePosY => Value::Builtin(Builtin::AbsolutePosY),
        Variable::AbsolutePosZ => Value::Builtin(Builtin::AbsolutePosZ),
    };
    Some(val)
}

fn id_of_op(op: &Operator) -> OpId {
    match op {
        Operator::Add(_) => OpId::Add,
        Operator::Fma(_) => OpId::Fma,
        Operator::Sub(_) => OpId::Sub,
        Operator::Mul(_) => OpId::Mul,
        Operator::Div(_) => OpId::Div,
        Operator::Abs(_) => OpId::Abs,
        Operator::Exp(_) => OpId::Exp,
        Operator::Log(_) => OpId::Log,
        Operator::Log1p(_) => OpId::Log1p,
        Operator::Cos(_) => OpId::Cos,
        Operator::Sin(_) => OpId::Sin,
        Operator::Tanh(_) => OpId::Tanh,
        Operator::Powf(_) => OpId::Powf,
        Operator::Sqrt(_) => OpId::Sqrt,
        Operator::Round(_) => OpId::Round,
        Operator::Floor(_) => OpId::Floor,
        Operator::Ceil(_) => OpId::Ceil,
        Operator::Erf(_) => OpId::Erf,
        Operator::Recip(_) => OpId::Recip,
        Operator::Equal(_) => OpId::Equal,
        Operator::NotEqual(_) => OpId::NotEqual,
        Operator::Lower(_) => OpId::Lower,
        Operator::Clamp(_) => OpId::Clamp,
        Operator::Greater(_) => OpId::Greater,
        Operator::LowerEqual(_) => OpId::LowerEqual,
        Operator::GreaterEqual(_) => OpId::GreaterEqual,
        Operator::Modulo(_) => OpId::Modulo,
        Operator::Index(_) => OpId::Index,
        Operator::UncheckedIndex(_) => OpId::Index,
        Operator::InitLine(_) => OpId::InitLine,
        Operator::And(_) => OpId::And,
        Operator::Or(_) => OpId::Or,
        Operator::Not(_) => OpId::Not,
        Operator::Neg(_) => OpId::Neg,
        Operator::Max(_) => OpId::Max,
        Operator::Min(_) => OpId::Min,
        Operator::BitwiseAnd(_) => OpId::BitwiseAnd,
        Operator::BitwiseOr(_) => OpId::BitwiseOr,
        Operator::BitwiseXor(_) => OpId::BitwiseXor,
        Operator::ShiftLeft(_) => OpId::ShiftLeft,
        Operator::ShiftRight(_) => OpId::ShiftRight,
        Operator::Remainder(_) => OpId::Remainder,
        Operator::Magnitude(_) => OpId::Magnitude,
        Operator::Normalize(_) => OpId::Normalize,
        Operator::Dot(_) => OpId::Dot,
        Operator::Bitcast(_) => OpId::Bitcast,
        _ => unreachable!(),
    }
}

fn id_of_meta(meta: &Metadata) -> OpId {
    match meta {
        Metadata::Stride { .. } => OpId::Stride,
        Metadata::Shape { .. } => OpId::Shape,
        Metadata::Length { .. } => OpId::Length,
    }
}

fn cmp_inverse(op: &OpId) -> OpId {
    match op {
        OpId::Lower => OpId::GreaterEqual,
        OpId::Greater => OpId::LowerEqual,
        OpId::LowerEqual => OpId::Greater,
        OpId::GreaterEqual => OpId::Lower,
        _ => unreachable!(),
    }
}

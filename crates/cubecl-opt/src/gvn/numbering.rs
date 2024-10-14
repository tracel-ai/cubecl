use std::{
    collections::{HashSet, LinkedList},
    mem::swap,
};

use cubecl_core::ir::{
    Branch, ConstantScalarValue, Elem, Item, Metadata, Operation, Operator, Subcube, Variable,
};

use crate::{passes::item_compatible, PhiInstruction};

use super::{
    convert::{id_of_meta, id_of_op, value_of_var},
    Expression, Instruction, OpId, Value, ValueTable,
};

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

    pub fn lookup_op(&mut self, op: &Operation) -> Option<u32> {
        let (expr, _) = self.create_expr(op).ok()?;
        self.expression_numbers.get(&expr).copied()
    }

    pub fn value_of_expr(&mut self, expr: Expression) -> (u32, bool) {
        if let Some(existing) = self.expression_numbers.get(&expr) {
            (*existing, false)
        } else if let Expression::Copy(num, _) = expr {
            self.expression_numbers.insert(expr.clone(), num);
            (num, true)
        } else {
            let num = self.next_value_num;
            self.expression_numbers.insert(expr.clone(), num);
            self.expressions.insert(self.next_expr_num, expr);
            self.next_value_num += 1;
            self.next_expr_num += 1;
            (num, true)
        }
    }

    pub fn lookup_or_add_phi(&mut self, phi: &PhiInstruction) -> (u32, Value) {
        let expr = Expression::Phi(
            phi.entries
                .iter()
                .map(|it| (value_of_var(&it.value).unwrap(), it.block))
                .collect(),
        );
        let out = value_of_var(&phi.out).unwrap();
        let num = self.lookup_or_add_expr(expr, Some(out));
        (num, out)
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
                let item = op.out.item();
                let cond = self.lookup_or_add_var(&op.cond)?;
                let then = self.lookup_or_add_var(&op.then)?;
                let or_else = self.lookup_or_add_var(&op.or_else)?;
                let expr = Instruction::new(OpId::Select, &[cond, then, or_else], item);
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
                let item = op.out.item();
                let is_cast = !item_compatible(op.input.item(), op.out.item());
                let out = value_of_var(&op.out);
                let num = self.lookup_or_add_var(&op.input)?;
                if is_cast {
                    let expr = Instruction::new(OpId::Cast, &[num], item);
                    (expr.into(), out)
                } else {
                    (Expression::Copy(num, item), out)
                }
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
                let item = op.out.item();
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let id = id_of_op(operator);
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                }
                let expr = Instruction::commutative(id, &[lhs, rhs], item);
                (expr.into(), out)
            }

            // Non-commutative binops
            Operator::Sub(op)
            | Operator::Div(op)
            | Operator::Powf(op)
            | Operator::Modulo(op)
            | Operator::Remainder(op)
            | Operator::ShiftLeft(op)
            | Operator::ShiftRight(op) => {
                let item = op.out.item();
                let lhs = self.lookup_or_add_var(&op.lhs)?;
                let rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let id = id_of_op(operator);
                let expr = Instruction::new(id, &[lhs, rhs], item);
                (expr.into(), out)
            }

            // Compare ops
            Operator::Lower(op)
            | Operator::Greater(op)
            | Operator::LowerEqual(op)
            | Operator::GreaterEqual(op) => {
                let item = op.out.item();
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&op.out);
                let mut op = id_of_op(operator);
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                    op = cmp_inverse(&op);
                }
                let expr = Instruction::new(op, &[lhs, rhs], item);
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
            | Operator::Magnitude(op)
            | Operator::Normalize(op) => {
                let input = self.lookup_or_add_var(&op.input)?;
                let item = op.out.item();
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &[input], item);
                (expr.into(), out)
            }
            Operator::Bitcast(op) => {
                let item = op.out.item();
                let input = self.lookup_or_add_var(&op.input)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &[input], item);
                (expr.into(), out)
            }

            Operator::Fma(op) => {
                let item = op.out.item();
                let mut a = self.lookup_or_add_var(&op.a)?;
                let mut b = self.lookup_or_add_var(&op.b)?;
                let c = self.lookup_or_add_var(&op.c)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                if a > b {
                    swap(&mut a, &mut b);
                }
                let expr = Instruction::new(op, &[a, b, c], item);
                (expr.into(), out)
            }
            Operator::Clamp(op) => {
                let item = op.out.item();
                let val = self.lookup_or_add_var(&op.input)?;
                let min = self.lookup_or_add_var(&op.min_value)?;
                let max = self.lookup_or_add_var(&op.max_value)?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &[val, min, max], item);
                (expr.into(), out)
            }
            Operator::InitLine(op) => {
                let item = op.out.item();
                let operands = op.inputs.iter().map(|it| self.lookup_or_add_var(it));
                let operands = operands.collect::<Result<Vec<_>, _>>()?;
                let out = value_of_var(&op.out);
                let op = id_of_op(operator);
                let expr = Instruction::new(op, &operands, item);
                (expr.into(), out)
            }

            Operator::Index(op) | Operator::UncheckedIndex(op) => {
                let out = value_of_var(&op.out);
                if !op.lhs.is_immutable() {
                    Err(out)?
                }
                let item = op.out.item();
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let id = id_of_op(operator);
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                }
                let expr = Instruction::commutative(id, &[lhs, rhs], item);
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
                let item = out.item();
                let var = self.lookup_or_add_var(var)?;
                let dim = self.lookup_or_add_var(dim)?;
                let out = value_of_var(out);
                let expr = Instruction::new(op, &[var, dim], item);
                (expr, out)
            }
            Metadata::Length { var, out } => {
                let item = out.item();
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
                        let expr = Expression::Copy(num, Item::new(Elem::UInt));
                        return Ok((expr, out));
                    }
                    _ => unreachable!("Length only available on array"),
                };
                let expr = Instruction::new(op, &[var], item);
                (expr, out)
            }
        };
        Ok((expr.into(), val))
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

use core::mem::swap;

use alloc::{collections::linked_list::LinkedList, vec::Vec};
use cubecl_ir::{
    self as ir, Arithmetic, AtomicOp, Comparison, ComparisonOpCode, Memory, OpCode, Operation,
    OperationReflect, Variable,
};
use hashbrown::HashSet;

use crate::PhiInstruction;

use super::{Expression, Instruction, Value, ValueTable, convert::value_of_var};

impl ValueTable {
    /// Look up or insert operation if it's numberable. Returns the number, optional out value and
    /// expression. If the error includes a value, treats that value as volatile (i.e. atomics) and
    /// don't number any expressions that depend on it.
    pub fn maybe_insert_op(
        &mut self,
        op: &ir::Instruction,
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

    pub fn lookup_op(&mut self, op: &ir::Instruction) -> Option<u32> {
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

    /// Create expression if it's numberable. Returns the number, optional out value and
    /// expression. If the error includes a value, treats that value as volatile (i.e. atomics) and
    /// don't number any expressions that depend on it.
    fn create_expr(
        &mut self,
        inst: &ir::Instruction,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        match &inst.operation {
            Operation::Copy(variable) => {
                let item = inst.ty();
                let out = value_of_var(&inst.out());
                let num = self.lookup_or_add_var(variable)?;
                Ok((Expression::Copy(num, item), out))
            }
            Operation::Memory(memory) => self.create_expr_memory(memory, inst.out),
            Operation::Arithmetic(arithmetic) => {
                self.create_expr_arithmetic(arithmetic, inst.out())
            }
            Operation::Comparison(cmp) => self.create_expr_cmp(cmp, inst.out()),
            Operation::Bitwise(bitwise) => self.create_expr_simple_op(bitwise, inst.out()),
            Operation::Operator(operator) => self.create_expr_simple_op(operator, inst.out()),
            Operation::Metadata(metadata) => self.create_expr_simple_op(metadata, inst.out()),
            Operation::Plane(_) => Err(value_of_var(&inst.out())),
            Operation::Atomic(atomic) => self.create_expr_atomic(atomic, inst.out),
            Operation::Branch(_)
            | Operation::Synchronization(_)
            | Operation::CoopMma(_)
            | Operation::NonSemantic(_)
            | Operation::Barrier(_)
            | Operation::Tma(_)
            | Operation::TensorIndexing(_)
            | Operation::Marker(_) => Err(None),
            Operation::ConstructAggregate(..) | Operation::ExtractAggregateField(..) => {
                unreachable!("Should be disaggregated at this point")
            }
        }
    }

    fn create_expr_arithmetic(
        &mut self,
        operator: &Arithmetic,
        out: Variable,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        let (expr, val) = match operator {
            Arithmetic::Fma(op) => {
                let item = out.ty;
                let mut a = self.lookup_or_add_var(&op.a)?;
                let mut b = self.lookup_or_add_var(&op.b)?;
                let c = self.lookup_or_add_var(&op.c)?;
                let out = value_of_var(&out);
                let op = OpCode::Arithmetic(operator.op_code());
                if a > b {
                    swap(&mut a, &mut b);
                }
                let expr = Instruction::new(op, &[a, b, c], item);
                (expr.into(), out)
            }

            op => self.create_expr_simple_op(op, out)?,
        };
        Ok((expr, val))
    }

    fn create_expr_cmp(
        &mut self,
        cmp: &Comparison,
        out: Variable,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        match cmp {
            // Compare ops
            Comparison::Lower(op)
            | Comparison::Greater(op)
            | Comparison::LowerEqual(op)
            | Comparison::GreaterEqual(op) => {
                let item = out.ty;
                let mut lhs = self.lookup_or_add_var(&op.lhs)?;
                let mut rhs = self.lookup_or_add_var(&op.rhs)?;
                let out = value_of_var(&out);
                let mut op = cmp.op_code();
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                    op = cmp_inverse(&op);
                }
                let expr = Instruction::new(op, &[lhs, rhs], item);
                Ok((expr.into(), out))
            }
            op => self.create_expr_simple_op(op, out),
        }
    }

    fn create_expr_memory(
        &mut self,
        memory: &Memory,
        out: Option<Variable>,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        let (expr, val) = match memory {
            Memory::Load(_) => Err(value_of_var(&out.unwrap()))?,
            Memory::Store(..) | Memory::CopyMemory(..) => Err(None)?,
            op => self.create_expr_simple_op(op, out.unwrap())?,
        };
        Ok((expr, val))
    }

    fn create_expr_atomic(
        &mut self,
        atomic: &AtomicOp,
        out: Option<Variable>,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        match atomic {
            AtomicOp::Store(..) => Err(None),
            _ => Err(value_of_var(&out.unwrap())),
        }
    }

    fn create_expr_simple_op<Code: Into<OpCode>>(
        &mut self,
        op: &impl OperationReflect<OpCode = Code>,
        out: Variable,
    ) -> Result<(Expression, Option<Value>), Option<Value>> {
        let item = out.ty;
        let id = op.op_code().into();
        let args = op.args();
        if let Some(args) = args {
            let mut args = args
                .iter()
                .map(|it| self.lookup_or_add_var(it))
                .collect::<Result<Vec<_>, _>>()?;
            let out = value_of_var(&out);
            let expr = if op.is_commutative() {
                args.sort();
                Instruction::commutative(id, &args, item)
            } else {
                Instruction::new(id, &args, item)
            };

            Ok((expr.into(), out))
        } else {
            Err(None)
        }
    }
}

fn cmp_inverse(op: &ComparisonOpCode) -> ComparisonOpCode {
    match op {
        ComparisonOpCode::Lower => ComparisonOpCode::Greater,
        ComparisonOpCode::Greater => ComparisonOpCode::Lower,
        ComparisonOpCode::LowerEqual => ComparisonOpCode::GreaterEqual,
        ComparisonOpCode::GreaterEqual => ComparisonOpCode::LowerEqual,
        _ => unreachable!(),
    }
}

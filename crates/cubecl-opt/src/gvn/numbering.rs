use core::mem::swap;

use alloc::{collections::linked_list::LinkedList, vec::Vec};
use cubecl_ir::{
    self as ir, Arithmetic, AtomicOp, Comparison, ComparisonOpCode, Memory, OpCode, Operation,
    OperationReflect, Operator, ExpandValue,
};
use hashbrown::HashSet;

use crate::{Function, PhiInstruction};

use super::{Expression, Instruction, ValueTable};

impl ValueTable {
    /// Look up or insert operation if it's numberable. Returns the number, optional out value and
    /// expression. If the error includes a value, treats that value as volatile (i.e. atomics) and
    /// don't number any expressions that depend on it.
    pub fn maybe_insert_op(
        &mut self,
        func: &Function,
        op: &ir::Instruction,
        exp_gen: &mut LinkedList<(u32, Expression)>,
        added_exps: &mut HashSet<u32>,
    ) -> Result<(u32, Option<ExpandValue>, Expression), Option<ExpandValue>> {
        let expr = self.create_expr(func, op);
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

    pub fn lookup_op(&mut self, func: &Function, op: &ir::Instruction) -> Option<u32> {
        let (expr, _) = self.create_expr(func, op).ok()?;
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

    pub fn lookup_or_add_phi(&mut self, func: &Function, phi: &PhiInstruction) -> (u32, ExpandValue) {
        let expr = Expression::Phi(
            phi.entries
                .iter()
                .map(|it| (func.value_of_var(&it.value).unwrap(), it.block))
                .collect(),
        );
        let out = func.value_of_var(&phi.out).unwrap();
        let num = self.lookup_or_add_expr(expr, Some(out));
        (num, out)
    }

    pub fn lookup_or_add_expr(&mut self, expr: Expression, value: Option<ExpandValue>) -> u32 {
        let num = self.value_of_expr(expr).0;
        if let Some(value) = value {
            self.value_numbers.insert(value, num);
            self.next_value_num += 1;
        }
        num
    }

    pub fn lookup_or_add_value(&mut self, value: ExpandValue) -> u32 {
        if let Some(existing) = self.value_numbers.get(&value) {
            *existing
        } else {
            let num = self.next_value_num;
            self.value_numbers.insert(value, num);
            self.next_value_num += 1;
            num
        }
    }

    pub fn lookup_or_add_var(
        &mut self,
        func: &Function,
        value: &ExpandValue,
    ) -> Result<u32, Option<ExpandValue>> {
        let value = func.value_of_var(value).ok_or(None)?;
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
        func: &Function,
        inst: &ir::Instruction,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        match &inst.operation {
            Operation::Copy(variable) => {
                let item = inst.ty();
                let out = func.value_of_var(&inst.out());
                let num = self.lookup_or_add_var(func, variable)?;
                Ok((Expression::Copy(num, item), out))
            }
            Operation::DeclareVariable { .. } => Err(None),
            Operation::Operator(Operator::ReadBuiltin(builtin)) => {
                let ty = inst.ty();
                let out = inst.out;
                Ok((Expression::Builtin(*builtin, ty), out))
            }
            Operation::Memory(memory) => self.create_expr_memory(func, memory, inst.out),
            Operation::Arithmetic(arithmetic) => {
                self.create_expr_arithmetic(func, arithmetic, inst.out())
            }
            Operation::Comparison(cmp) => self.create_expr_cmp(func, cmp, inst.out()),
            Operation::Bitwise(bitwise) => self.create_expr_simple_op(func, bitwise, inst.out()),
            Operation::Operator(operator) => self.create_expr_simple_op(func, operator, inst.out()),
            Operation::Metadata(metadata) => self.create_expr_simple_op(func, metadata, inst.out()),
            // Not numberable: it has a barrier side-effect and a
            // workgroup-uniformity contract, so it must never be deduplicated.
            Operation::Plane(_) | Operation::WorkgroupUniformLoad(_) => {
                Err(func.value_of_var(&inst.out()))
            }
            Operation::Atomic(atomic) => self.create_expr_atomic(func, atomic, inst.out),
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
        func: &Function,
        operator: &Arithmetic,
        out: ExpandValue,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        let (expr, val) = match operator {
            Arithmetic::Fma(op) => {
                let item = out.ty;
                let mut a = self.lookup_or_add_var(func, &op.a)?;
                let mut b = self.lookup_or_add_var(func, &op.b)?;
                let c = self.lookup_or_add_var(func, &op.c)?;
                let out = func.value_of_var(&out);
                let op = OpCode::Arithmetic(operator.op_code());
                if a > b {
                    swap(&mut a, &mut b);
                }
                let expr = Instruction::new(op, &[a, b, c], item);
                (expr.into(), out)
            }

            op => self.create_expr_simple_op(func, op, out)?,
        };
        Ok((expr, val))
    }

    fn create_expr_cmp(
        &mut self,
        func: &Function,
        cmp: &Comparison,
        out: ExpandValue,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        match cmp {
            // Compare ops
            Comparison::Lower(op)
            | Comparison::Greater(op)
            | Comparison::LowerEqual(op)
            | Comparison::GreaterEqual(op) => {
                let item = out.ty;
                let mut lhs = self.lookup_or_add_var(func, &op.lhs)?;
                let mut rhs = self.lookup_or_add_var(func, &op.rhs)?;
                let out = func.value_of_var(&out);
                let mut op = cmp.op_code();
                if lhs > rhs {
                    swap(&mut lhs, &mut rhs);
                    op = cmp_inverse(&op);
                }
                let expr = Instruction::new(op, &[lhs, rhs], item);
                Ok((expr.into(), out))
            }
            op => self.create_expr_simple_op(func, op, out),
        }
    }

    fn create_expr_memory(
        &mut self,
        func: &Function,
        memory: &Memory,
        out: Option<ExpandValue>,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        let (expr, val) = match memory {
            Memory::Load(_) => Err(func.value_of_var(&out.unwrap()))?,
            Memory::Store(..) | Memory::CopyMemory(..) => Err(None)?,
            op => self.create_expr_simple_op(func, op, out.unwrap())?,
        };
        Ok((expr, val))
    }

    fn create_expr_atomic(
        &mut self,
        func: &Function,
        atomic: &AtomicOp,
        out: Option<ExpandValue>,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        match atomic {
            AtomicOp::Store(..) => Err(None),
            _ => Err(func.value_of_var(&out.unwrap())),
        }
    }

    fn create_expr_simple_op<Code: Into<OpCode>>(
        &mut self,
        func: &Function,
        op: &impl OperationReflect<OpCode = Code>,
        out: ExpandValue,
    ) -> Result<(Expression, Option<ExpandValue>), Option<ExpandValue>> {
        let item = out.ty;
        let id = op.op_code().into();
        let args = op.args();
        if let Some(args) = args {
            let mut args = args
                .iter()
                .map(|it| self.lookup_or_add_var(func, it))
                .collect::<Result<Vec<_>, _>>()?;
            let out = func.value_of_var(&out);
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

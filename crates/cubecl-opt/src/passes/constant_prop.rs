use cubecl_ir::{
    Arithmetic, Bitwise, Comparison, ConstantScalarValue, Instruction, Metadata, Operation,
    Operator, UIntKind, Variable, VariableKind,
};

use crate::{
    AtomicCounter, Optimizer,
    analyses::const_len::{Slice, Slices},
};

use super::OptimizerPass;

/// Simplifies certain expressions where one operand is constant.
/// For example: `out = x * 1` to `out = x`
pub struct ConstOperandSimplify;

impl OptimizerPass for ConstOperandSimplify {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let slices = opt.analysis::<Slices>();

        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

            for idx in ops {
                let op = &mut opt.program[node].ops.borrow_mut()[idx];
                match &mut op.operation {
                    Operation::Arithmetic(operator) => match operator {
                        // 0 * x == 0
                        Arithmetic::Mul(bin_op) if bin_op.lhs.is_constant(0) => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // x * 0 == 0
                        Arithmetic::Mul(bin_op) if bin_op.rhs.is_constant(0) => {
                            op.operation = Operation::Copy(bin_op.rhs);
                            changes.inc();
                        }
                        // 1 * x == x
                        Arithmetic::Mul(bin_op) if bin_op.lhs.is_constant(1) => {
                            op.operation = Operation::Copy(bin_op.rhs);
                            changes.inc();
                        }
                        // x * 1 == x
                        Arithmetic::Mul(bin_op) if bin_op.rhs.is_constant(1) => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // 0 + x = x
                        Arithmetic::Add(bin_op) if bin_op.lhs.is_constant(0) => {
                            op.operation = Operation::Copy(bin_op.rhs);
                            changes.inc();
                        }
                        // x + 0 = x
                        Arithmetic::Add(bin_op) if bin_op.rhs.is_constant(0) => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // x - 0 == x
                        Arithmetic::Sub(bin_op) if bin_op.rhs.is_constant(0) => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // x / 1 == x, 0 / x == 0
                        Arithmetic::Div(bin_op)
                            if bin_op.lhs.is_constant(0) || bin_op.rhs.is_constant(1) =>
                        {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // x % 1 == 0, 0 % x == 0
                        Arithmetic::Modulo(bin_op)
                            if bin_op.rhs.is_constant(1) || bin_op.lhs.is_constant(0) =>
                        {
                            let value = ConstantScalarValue::UInt(0, UIntKind::U32)
                                .cast_to(op.item().elem());
                            op.operation = Operation::Copy(Variable::constant(value));
                            changes.inc();
                        }
                        _ => {}
                    },

                    Operation::Operator(operator) => match operator {
                        // true || x == true, x || true == true
                        Operator::Or(bin_op) if bin_op.lhs.is_true() || bin_op.rhs.is_true() => {
                            op.operation = Operation::Copy(true.into());
                            changes.inc();
                        }
                        // false || x == x, x || false == x
                        Operator::Or(bin_op) if bin_op.lhs.is_false() => {
                            op.operation = Operation::Copy(bin_op.rhs);
                            changes.inc();
                        }
                        // x || false == x
                        Operator::Or(bin_op) if bin_op.rhs.is_false() => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // false && x == false, x && false == false
                        Operator::And(bin_op) if bin_op.lhs.is_false() || bin_op.rhs.is_false() => {
                            op.operation = Operation::Copy(false.into());
                            changes.inc();
                        }
                        // true && x == x
                        Operator::And(bin_op) if bin_op.lhs.is_true() => {
                            op.operation = Operation::Copy(bin_op.rhs);
                            changes.inc();
                        }
                        // x && true == x
                        Operator::And(bin_op) if bin_op.rhs.is_true() => {
                            op.operation = Operation::Copy(bin_op.lhs);
                            changes.inc();
                        }
                        // select(true, a, b) == a
                        Operator::Select(select) if select.cond.is_true() => {
                            op.operation = Operation::Copy(select.then);
                            changes.inc();
                        }
                        // select(false, a, b) == b
                        Operator::Select(select) if select.cond.is_false() => {
                            op.operation = Operation::Copy(select.or_else);
                            changes.inc();
                        }
                        _ => {}
                    },

                    // Constant length to const value
                    Operation::Metadata(Metadata::Length { var }) => match var.kind {
                        VariableKind::ConstantArray { length, .. }
                        | VariableKind::SharedMemory { length, .. }
                        | VariableKind::LocalArray { length, .. } => {
                            op.operation = Operation::Copy(length.into());
                            changes.inc();
                        }
                        VariableKind::Slice { id } => {
                            let slice = slices.get(&id);
                            if let Some(Slice {
                                const_len: Some(len),
                                ..
                            }) = slice
                            {
                                op.operation = Operation::Copy((*len).into());
                                changes.inc();
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
}

/// Evaluates expressions where both operands are constant and replaces them with simple constant
/// assignments. This can often be applied as a result of assignment merging or constant operand
/// simplification.
pub struct ConstEval;

impl OptimizerPass for ConstEval {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.node_ids() {
            let ops = opt.program[node].ops.clone();
            for op in ops.borrow_mut().values_mut() {
                if let Some(const_eval) = try_const_eval(op) {
                    let input = Variable::constant(const_eval);
                    op.operation = Operation::Copy(input);
                    changes.inc();
                }
            }
        }
    }
}

macro_rules! const_eval {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs $op rhs, kind),
                (Float(lhs, kind), Float(rhs, _)) => ConstantScalarValue::Float(lhs $op rhs, kind),
                (UInt(lhs, kind), UInt(rhs, _)) => ConstantScalarValue::UInt(lhs $op rhs, kind),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($lhs:expr, $rhs:expr; $op:path) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int($op(lhs, rhs), kind),
                (Float(lhs, kind), Float(rhs, _)) => ConstantScalarValue::Float($op(lhs, rhs), kind),
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt($op(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_int {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs $op rhs, kind),
                (UInt(lhs, kind), UInt(rhs, _)) => ConstantScalarValue::UInt(lhs $op rhs, kind),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_float {
    ($lhs:expr, $rhs:expr; $fn:path) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(lhs.elem());
            Some(match (lhs, rhs) {
                (Float(lhs, kind), Float(rhs, _)) => {
                    ConstantScalarValue::Float($fn(lhs, rhs), kind)
                }
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($input:expr; $fn:path) => {{
        use ConstantScalarValue::*;

        if let Some(input) = $input.as_const() {
            Some(match input {
                Float(input, kind) => ConstantScalarValue::Float($fn(input), kind),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_cmp {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Int(lhs, _), Int(rhs, _)) => ConstantScalarValue::Bool(lhs $op rhs),
                (Float(lhs, _), Float(rhs, _)) => ConstantScalarValue::Bool(lhs $op rhs),
                (UInt(lhs, _), UInt(rhs, _)) => ConstantScalarValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_bool {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantScalarValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Bool(lhs), Bool(rhs)) => ConstantScalarValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

fn try_const_eval(inst: &mut Instruction) -> Option<ConstantScalarValue> {
    match &mut inst.operation {
        Operation::Arithmetic(op) => try_const_eval_arithmetic(op),
        Operation::Comparison(op) => try_const_eval_cmp(op),
        Operation::Bitwise(op) => try_const_eval_bitwise(op),
        Operation::Operator(op) => try_const_eval_operator(op),
        _ => None,
    }
}

fn try_const_eval_arithmetic(op: &mut Arithmetic) -> Option<ConstantScalarValue> {
    match op {
        Arithmetic::Add(op) => const_eval!(+ op.lhs, op.rhs),
        Arithmetic::Sub(op) => const_eval!(-op.lhs, op.rhs),
        Arithmetic::Mul(op) => const_eval!(*op.lhs, op.rhs),
        Arithmetic::Div(op) => const_eval!(/ op.lhs, op.rhs),
        Arithmetic::Powf(op) => const_eval_float!(op.lhs, op.rhs; num::Float::powf),
        Arithmetic::Modulo(op) => const_eval!(% op.lhs, op.rhs),
        Arithmetic::Remainder(op) => const_eval!(% op.lhs, op.rhs),
        Arithmetic::Max(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.max(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.max(rhs), kind)
                    }
                    (UInt(lhs, kind), UInt(rhs, _)) => {
                        ConstantScalarValue::UInt(lhs.max(rhs), kind)
                    }
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Min(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.min(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.min(rhs), kind)
                    }
                    (UInt(lhs, kind), UInt(rhs, _)) => {
                        ConstantScalarValue::UInt(lhs.min(rhs), kind)
                    }
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Dot(op) => const_eval!(*op.lhs, op.rhs),

        Arithmetic::Abs(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(input.abs(), kind),
                Float(input, kind) => ConstantScalarValue::Float(input.abs(), kind),
                _ => unreachable!(),
            })
        }
        Arithmetic::Exp(op) => const_eval_float!(op.input; num::Float::exp),
        Arithmetic::Log(op) => const_eval_float!(op.input; num::Float::ln),
        Arithmetic::Log1p(op) => const_eval_float!(op.input; num::Float::ln_1p),
        Arithmetic::Cos(op) => const_eval_float!(op.input; num::Float::cos),
        Arithmetic::Sin(op) => const_eval_float!(op.input; num::Float::sin),
        Arithmetic::Tanh(op) => const_eval_float!(op.input; num::Float::tanh),
        Arithmetic::Sqrt(op) => const_eval_float!(op.input; num::Float::sqrt),
        Arithmetic::Round(op) => const_eval_float!(op.input; num::Float::round),
        Arithmetic::Floor(op) => const_eval_float!(op.input; num::Float::floor),
        Arithmetic::Ceil(op) => const_eval_float!(op.input; num::Float::ceil),
        Arithmetic::Recip(op) => const_eval_float!(op.input; num::Float::recip),
        Arithmetic::Neg(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(-input, kind),
                Float(input, kind) => ConstantScalarValue::Float(-input, kind),
                _ => unreachable!(),
            })
        }

        Arithmetic::Fma(op) => {
            use ConstantScalarValue::*;
            let a = op.a.as_const();
            let b = op.b.as_const();
            let c = op.c.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(a.elem());
                let c = c.cast_to(a.elem());
                match (a, b, c) {
                    (Float(a, kind), Float(b, _), Float(c, _)) => {
                        ConstantScalarValue::Float(a * b + c, kind)
                    }
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Clamp(op) => {
            use ConstantScalarValue::*;
            let a = op.input.as_const();
            let b = op.min_value.as_const();
            let c = op.max_value.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(a.elem());
                let c = c.cast_to(a.elem());
                match (a, b, c) {
                    (Int(a, kind), Int(b, _), Int(c, _)) => {
                        ConstantScalarValue::Int(a.clamp(b, c), kind)
                    }
                    (Float(a, kind), Float(b, _), Float(c, _)) => {
                        ConstantScalarValue::Float(a.clamp(b, c), kind)
                    }
                    (UInt(a, kind), UInt(b, _), UInt(c, _)) => {
                        ConstantScalarValue::UInt(a.clamp(b, c), kind)
                    }
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Erf(_) | Arithmetic::Magnitude(_) | Arithmetic::Normalize(_) => None,
    }
}

fn try_const_eval_cmp(op: &mut Comparison) -> Option<ConstantScalarValue> {
    match op {
        Comparison::Equal(op) => const_eval_cmp!(== op.lhs, op.rhs),
        Comparison::NotEqual(op) => const_eval_cmp!(!= op.lhs, op.rhs),
        Comparison::Lower(op) => const_eval_cmp!(< op.lhs, op.rhs),
        Comparison::Greater(op) => const_eval_cmp!(> op.lhs, op.rhs),
        Comparison::LowerEqual(op) => const_eval_cmp!(<= op.lhs, op.rhs),
        Comparison::GreaterEqual(op) => const_eval_cmp!(>= op.lhs, op.rhs),
    }
}

fn try_const_eval_bitwise(op: &mut Bitwise) -> Option<ConstantScalarValue> {
    match op {
        Bitwise::BitwiseAnd(op) => const_eval_int!(&op.lhs, op.rhs),
        Bitwise::BitwiseOr(op) => const_eval_int!(| op.lhs, op.rhs),
        Bitwise::BitwiseXor(op) => const_eval_int!(^ op.lhs, op.rhs),
        Bitwise::ShiftLeft(op) => const_eval_int!(<< op.lhs, op.rhs),
        Bitwise::ShiftRight(op) => const_eval_int!(>> op.lhs, op.rhs),
        Bitwise::BitwiseNot(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(!input, kind),
                UInt(input, kind) => ConstantScalarValue::UInt(!input, kind),
                _ => unreachable!(),
            })
        }
        Bitwise::CountOnes(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(input.count_ones() as i64, kind),
                UInt(input, kind) => ConstantScalarValue::UInt(input.count_ones() as u64, kind),
                _ => unreachable!(),
            })
        }
        Bitwise::ReverseBits(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(input.reverse_bits(), kind),
                UInt(input, kind) => ConstantScalarValue::UInt(input.reverse_bits(), kind),
                _ => unreachable!(),
            })
        }
        Bitwise::LeadingZeros(_) | Bitwise::FindFirstSet(_) => {
            // Depends too much on type width and Rust semantics, leave this one out of const eval
            None
        }
    }
}

fn try_const_eval_operator(op: &mut Operator) -> Option<ConstantScalarValue> {
    match op {
        Operator::And(op) => const_eval_bool!(&&op.lhs, op.rhs),
        Operator::Or(op) => const_eval_bool!(|| op.lhs, op.rhs),
        Operator::Not(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Bool(input) => ConstantScalarValue::Bool(!input),
                _ => unreachable!(),
            })
        }
        Operator::Cast(_)
        | Operator::Index(_)
        | Operator::CopyMemory(_)
        | Operator::CopyMemoryBulk(_)
        | Operator::Slice(_)
        | Operator::UncheckedIndex(_)
        | Operator::IndexAssign(_)
        | Operator::InitLine(_)
        | Operator::UncheckedIndexAssign(_)
        | Operator::Bitcast(_)
        | Operator::Select(_)
        | Operator::ConditionalRead(_) => None,
    }
}

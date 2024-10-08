use cubecl_core::ir::{Branch, ConstantScalarValue, Operation, Operator, UnaryOperator, Variable};

use crate::{AtomicCounter, Optimizer};

use super::{get_out, OptimizerPass};

/// Simplifies certain expressions where one operand is constant.
/// For example: `out = x * 1` to `out = x`
pub struct ConstOperandSimplify;

impl OptimizerPass for ConstOperandSimplify {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

            for idx in ops {
                let op = &mut opt.program[node].ops.borrow_mut()[idx];
                match op {
                    Operation::Operator(Operator::Mul(bin_op)) => {
                        let value = match (bin_op.lhs.as_const(), bin_op.rhs.as_const()) {
                            // 0 * x == 0, x * 0 == 0
                            (Some(val), _) | (_, Some(val)) if val.is_zero() => {
                                Some(Variable::ConstantScalar(val))
                            }
                            // 1 * x == x
                            (Some(lhs), _) if lhs.is_one() => Some(bin_op.rhs),
                            // x * 1 == x
                            (_, Some(rhs)) if rhs.is_one() => Some(bin_op.lhs),
                            _ => None,
                        };
                        if let Some(value) = value {
                            *op = assign(bin_op.out, value);
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Add(bin_op)) => {
                        let value = match (bin_op.lhs.as_const(), bin_op.rhs.as_const()) {
                            // x + 0 == x
                            (Some(lhs), _) if lhs.is_zero() => Some(bin_op.rhs),
                            // 0 + x == x
                            (_, Some(rhs)) if rhs.is_zero() => Some(bin_op.lhs),
                            _ => None,
                        };
                        if let Some(value) = value {
                            *op = assign(bin_op.out, value);
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Sub(bin_op)) => {
                        let rhs = bin_op.rhs.as_const();
                        let rhs_zero = rhs.map(|it| it.is_zero()).unwrap_or(false);
                        // x - 0 == x
                        if rhs_zero {
                            *op = assign(bin_op.out, bin_op.lhs);
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Div(bin_op)) => {
                        let lhs = bin_op.lhs.as_const();
                        let lhs_zero = lhs.map(|it| it.is_zero()).unwrap_or(false);
                        let rhs_one = bin_op.rhs.as_const().map(|it| it.is_one()).unwrap_or(false);
                        // x / 1 == x, 0 / x == 0
                        if rhs_one || lhs_zero {
                            *op = assign(bin_op.out, bin_op.lhs);
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::Or(bin_op)) => {
                        let lhs = bin_op.lhs.as_const();
                        let rhs = bin_op.rhs.as_const();
                        let assign_val = match (lhs, rhs) {
                            // true || x == true, x || true == true
                            (Some(ConstantScalarValue::Bool(true)), _)
                            | (_, Some(ConstantScalarValue::Bool(true))) => {
                                Some(Variable::ConstantScalar(ConstantScalarValue::Bool(true)))
                            }
                            // false || x == x
                            (Some(ConstantScalarValue::Bool(false)), _) => Some(bin_op.rhs),
                            // x || false == x
                            (_, Some(ConstantScalarValue::Bool(false))) => Some(bin_op.lhs),
                            _ => None,
                        };
                        if let Some(assign_val) = assign_val {
                            *op = assign(bin_op.out, assign_val);
                            changes.inc();
                        }
                    }
                    Operation::Operator(Operator::And(bin_op)) => {
                        let lhs = bin_op.lhs.as_const();
                        let rhs = bin_op.rhs.as_const();
                        let assign_val = match (lhs, rhs) {
                            // true && x == x
                            (Some(ConstantScalarValue::Bool(true)), _) => Some(bin_op.rhs),
                            // x && true == x
                            (_, Some(ConstantScalarValue::Bool(true))) => Some(bin_op.lhs),
                            // false && x == false, x && false == false
                            (Some(ConstantScalarValue::Bool(false)), _)
                            | (_, Some(ConstantScalarValue::Bool(false))) => {
                                Some(Variable::ConstantScalar(ConstantScalarValue::Bool(false)))
                            }
                            _ => None,
                        };
                        if let Some(assign_val) = assign_val {
                            *op = assign(bin_op.out, assign_val);
                            changes.inc();
                        }
                    }
                    Operation::Branch(Branch::Select(select)) => {
                        if let Some(cond) = select.cond.as_const() {
                            let value = match cond.as_bool() {
                                // select(true, a, b) == a
                                true => select.then,
                                // select(false, a, b) == b
                                false => select.or_else,
                            };
                            *op = assign(select.out, value);
                            changes.inc();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn assign(out: Variable, value: Variable) -> Operation {
    Operation::Operator(Operator::Assign(UnaryOperator { input: value, out }))
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
                let out = get_out(opt, op);
                if let Some(out) = out {
                    if let Some(const_eval) = try_const_eval(op) {
                        let input = Variable::ConstantScalar(const_eval);
                        *op = Operator::Assign(UnaryOperator { out, input }).into();
                        changes.inc();
                    }
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
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs $op rhs),
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
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs $op rhs),
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
                (UInt(lhs), UInt(rhs)) => ConstantScalarValue::Bool(lhs $op rhs),
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

fn try_const_eval(op: &mut Operation) -> Option<ConstantScalarValue> {
    let op = match op {
        Operation::Operator(operator) => operator,
        _ => return None,
    };

    match op {
        Operator::Add(op) => const_eval!(+ op.lhs, op.rhs),
        Operator::Sub(op) => const_eval!(-op.lhs, op.rhs),
        Operator::Mul(op) => const_eval!(*op.lhs, op.rhs),
        Operator::Div(op) => const_eval!(/ op.lhs, op.rhs),
        Operator::Powf(op) => const_eval_float!(op.lhs, op.rhs; num::Float::powf),
        Operator::Equal(op) => const_eval_cmp!(== op.lhs, op.rhs),
        Operator::NotEqual(op) => const_eval_cmp!(!= op.lhs, op.rhs),
        Operator::Lower(op) => const_eval_cmp!(< op.lhs, op.rhs),
        Operator::Greater(op) => const_eval_cmp!(> op.lhs, op.rhs),
        Operator::LowerEqual(op) => const_eval_cmp!(<= op.lhs, op.rhs),
        Operator::GreaterEqual(op) => const_eval_cmp!(>= op.lhs, op.rhs),
        Operator::Modulo(op) => const_eval!(% op.lhs, op.rhs),
        Operator::And(op) => const_eval_bool!(&&op.lhs, op.rhs),
        Operator::Or(op) => const_eval_bool!(|| op.lhs, op.rhs),
        Operator::Max(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.max(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.max(rhs), kind)
                    }
                    (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs.max(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Operator::Min(op) => {
            use ConstantScalarValue::*;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(lhs.elem());
                Some(match (lhs, rhs) {
                    (Int(lhs, kind), Int(rhs, _)) => ConstantScalarValue::Int(lhs.min(rhs), kind),
                    (Float(lhs, kind), Float(rhs, _)) => {
                        ConstantScalarValue::Float(lhs.min(rhs), kind)
                    }
                    (UInt(lhs), UInt(rhs)) => ConstantScalarValue::UInt(lhs.min(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Operator::BitwiseAnd(op) => const_eval_int!(&op.lhs, op.rhs),
        Operator::BitwiseOr(op) => const_eval_int!(| op.lhs, op.rhs),
        Operator::BitwiseXor(op) => const_eval_int!(^ op.lhs, op.rhs),
        Operator::ShiftLeft(op) => const_eval_int!(<< op.lhs, op.rhs),
        Operator::ShiftRight(op) => const_eval_int!(>> op.lhs, op.rhs),
        Operator::Dot(op) => const_eval!(*op.lhs, op.rhs),

        Operator::Abs(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(input.abs(), kind),
                Float(input, kind) => ConstantScalarValue::Float(input.abs(), kind),
                _ => unreachable!(),
            })
        }
        Operator::Exp(op) => const_eval_float!(op.input; num::Float::exp),
        Operator::Log(op) => const_eval_float!(op.input; num::Float::ln),
        Operator::Log1p(op) => const_eval_float!(op.input; num::Float::ln_1p),
        Operator::Cos(op) => const_eval_float!(op.input; num::Float::cos),
        Operator::Sin(op) => const_eval_float!(op.input; num::Float::sin),
        Operator::Tanh(op) => const_eval_float!(op.input; num::Float::tanh),
        Operator::Sqrt(op) => const_eval_float!(op.input; num::Float::sqrt),
        Operator::Round(op) => const_eval_float!(op.input; num::Float::round),
        Operator::Floor(op) => const_eval_float!(op.input; num::Float::floor),
        Operator::Ceil(op) => const_eval_float!(op.input; num::Float::ceil),
        Operator::Recip(op) => const_eval_float!(op.input; num::Float::recip),
        Operator::Not(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Bool(input) => ConstantScalarValue::Bool(!input),
                _ => unreachable!(),
            })
        }
        Operator::Neg(op) => {
            use ConstantScalarValue::*;
            op.input.as_const().map(|input| match input {
                Int(input, kind) => ConstantScalarValue::Int(-input, kind),
                Float(input, kind) => ConstantScalarValue::Float(-input, kind),
                _ => unreachable!(),
            })
        }

        Operator::Fma(op) => {
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
        Operator::Clamp(op) => {
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
                    (UInt(a), UInt(b), UInt(c)) => ConstantScalarValue::UInt(a.clamp(b, c)),
                    _ => unreachable!(),
                }
            })
        }
        _ => None,
    }
}

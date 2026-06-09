use alloc::{vec, vec::Vec};
use cubecl_ir::{
    Arithmetic, Bitwise, Comparison, ConstantValue, GlobalState, Instruction, Operation, Operator,
    Type, Variable,
};

use crate::post_processing::{
    analysis_helper::GlobalAnalyses, util::AtomicCounter, visitor::InstructionVisitor,
};

/// Simplifies certain expressions where one operand is constant.
/// For example: `out = x * 1` to `out = x`
#[derive(Debug)]
pub struct ConstOperandSimplify;

impl InstructionVisitor for ConstOperandSimplify {
    fn visit_instruction(
        &mut self,
        mut inst: Instruction,
        _state: &GlobalState,
        _analyses: &GlobalAnalyses,
        changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        match &mut inst.operation {
            Operation::Arithmetic(operator) => match operator {
                // 0 * x == 0
                Arithmetic::Mul(bin_op) if bin_op.lhs.is_constant(0) => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // x * 0 == 0
                Arithmetic::Mul(bin_op) if bin_op.rhs.is_constant(0) => {
                    inst.operation = Operation::Copy(bin_op.rhs);
                    changes.inc();
                }
                // 1 * x == x
                Arithmetic::Mul(bin_op) if bin_op.lhs.is_constant(1) => {
                    inst.operation = Operation::Copy(bin_op.rhs);
                    changes.inc();
                }
                // x * 1 == x
                Arithmetic::Mul(bin_op) if bin_op.rhs.is_constant(1) => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // 0 + x = x
                Arithmetic::Add(bin_op) if bin_op.lhs.is_constant(0) => {
                    inst.operation = Operation::Copy(bin_op.rhs);
                    changes.inc();
                }
                // x + 0 = x
                Arithmetic::Add(bin_op) if bin_op.rhs.is_constant(0) => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // x - 0 == x
                Arithmetic::Sub(bin_op) if bin_op.rhs.is_constant(0) => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // x / 1 == x, 0 / x == 0
                Arithmetic::Div(bin_op)
                    if bin_op.lhs.is_constant(0) || bin_op.rhs.is_constant(1) =>
                {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // x % 1 == 0, 0 % x == 0
                Arithmetic::ModFloor(bin_op) | Arithmetic::Rem(bin_op)
                    if bin_op.rhs.is_constant(1) || bin_op.lhs.is_constant(0) =>
                {
                    inst.operation = Operation::Copy(inst.ty().constant(ConstantValue::Int(0)));
                    changes.inc();
                }
                _ => {}
            },

            Operation::Operator(operator) => match operator {
                // true || x == true, x || true == true
                Operator::Or(bin_op) if bin_op.lhs.is_true() || bin_op.rhs.is_true() => {
                    inst.operation = Operation::Copy(true.into());
                    changes.inc();
                }
                // false || x == x, x || false == x
                Operator::Or(bin_op) if bin_op.lhs.is_false() => {
                    inst.operation = Operation::Copy(bin_op.rhs);
                    changes.inc();
                }
                // x || false == x
                Operator::Or(bin_op) if bin_op.rhs.is_false() => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // false && x == false, x && false == false
                Operator::And(bin_op) if bin_op.lhs.is_false() || bin_op.rhs.is_false() => {
                    inst.operation = Operation::Copy(false.into());
                    changes.inc();
                }
                // true && x == x
                Operator::And(bin_op) if bin_op.lhs.is_true() => {
                    inst.operation = Operation::Copy(bin_op.rhs);
                    changes.inc();
                }
                // x && true == x
                Operator::And(bin_op) if bin_op.rhs.is_true() => {
                    inst.operation = Operation::Copy(bin_op.lhs);
                    changes.inc();
                }
                // select(true, a, b) == a
                Operator::Select(select) if select.cond.is_true() => {
                    inst.operation = Operation::Copy(select.then);
                    changes.inc();
                }
                // select(false, a, b) == b
                Operator::Select(select) if select.cond.is_false() => {
                    inst.operation = Operation::Copy(select.or_else);
                    changes.inc();
                }
                _ => {}
            },
            _ => {}
        };
        vec![inst]
    }
}

/// Evaluates expressions where both operands are constant and replaces them with simple constant
/// assignments. This can often be applied as a result of assignment merging or constant operand
/// simplification.
#[derive(Debug)]
pub struct ConstEval;

impl InstructionVisitor for ConstEval {
    fn visit_instruction(
        &mut self,
        mut inst: Instruction,
        _state: &GlobalState,
        _analyses: &GlobalAnalyses,
        changes: &AtomicCounter,
    ) -> Vec<Instruction> {
        if let Some(const_eval) = try_const_eval(&mut inst) {
            let input = Variable::constant(const_eval, inst.out().ty);
            inst.operation = Operation::Copy(input);
            changes.inc();
        }
        vec![inst]
    }
}

macro_rules! const_eval {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs $op rhs),
                (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_int {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => Int(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => UInt(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($lhs:expr, $rhs:expr; $op:path) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => Int($op(lhs, rhs)),
                (UInt(lhs), UInt(rhs)) => UInt($op(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_float {
    ($lhs:expr, $rhs:expr; $fn:path) => {{
        use ConstantValue::*;

        let ty = $lhs.ty;
        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            let rhs = rhs.cast_to(ty);
            Some(match (lhs, rhs) {
                (Float(lhs), Float(rhs)) => ConstantValue::Float($fn(lhs, rhs)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
    ($input:expr; $fn:path) => {{
        use ConstantValue::*;

        if let Some(input) = $input.as_const() {
            Some(match input {
                Float(input) => ConstantValue::Float($fn(input)),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_cmp {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Int(lhs), Int(rhs)) => ConstantValue::Bool(lhs $op rhs),
                (Float(lhs), Float(rhs)) => ConstantValue::Bool(lhs $op rhs),
                (UInt(lhs), UInt(rhs)) => ConstantValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

macro_rules! const_eval_bool {
    ($op:tt $lhs:expr, $rhs:expr) => {{
        use ConstantValue::*;

        let lhs = $lhs.as_const();
        let rhs = $rhs.as_const();
        if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
            Some(match (lhs, rhs) {
                (Bool(lhs), Bool(rhs)) => ConstantValue::Bool(lhs $op rhs),
                _ => unreachable!(),
            })
        } else {
            None
        }
    }};
}

fn try_const_eval(inst: &mut Instruction) -> Option<ConstantValue> {
    match &mut inst.operation {
        Operation::Arithmetic(op) => try_const_eval_arithmetic(op),
        Operation::Comparison(op) => try_const_eval_cmp(op),
        Operation::Bitwise(op) => try_const_eval_bitwise(op),
        Operation::Operator(op) => try_const_eval_operator(op, inst.out.map(|it| it.ty)),
        _ => None,
    }
}

fn try_const_eval_arithmetic(op: &mut Arithmetic) -> Option<ConstantValue> {
    match op {
        Arithmetic::Add(op) => const_eval!(+ op.lhs, op.rhs),
        Arithmetic::Sub(op) => const_eval!(-op.lhs, op.rhs),
        Arithmetic::Mul(op) => const_eval!(*op.lhs, op.rhs),
        Arithmetic::Div(op) => const_eval!(/ op.lhs, op.rhs),
        Arithmetic::SaturatingAdd(op) => {
            use ConstantValue::*;

            let ty = op.lhs.ty;
            let lhs = op.lhs.as_const();
            let rhs = op.rhs.as_const();
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                let rhs = rhs.cast_to(ty);
                let width = ty.size();
                Some(match (lhs, rhs, width) {
                    (Int(lhs), Int(rhs), 4) => Int((lhs as i32).saturating_add(rhs as i32) as i64),
                    (Int(lhs), Int(rhs), 8) => Int(lhs.saturating_add(rhs)),
                    (UInt(lhs), UInt(rhs), 4) => {
                        UInt((lhs as u32).saturating_add(rhs as u32) as u64)
                    }
                    (UInt(lhs), UInt(rhs), 8) => UInt(lhs.saturating_add(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::SaturatingSub(op) => {
            use ConstantValue::*;

            let ty = op.lhs.ty;
            let lhs = op.lhs.as_const();
            let rhs = op.rhs.as_const();
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                let rhs = rhs.cast_to(ty);
                let width = ty.size();
                Some(match (lhs, rhs, width) {
                    (Int(lhs), Int(rhs), 4) => Int((lhs as i32).saturating_sub(rhs as i32) as i64),
                    (Int(lhs), Int(rhs), 8) => Int(lhs.saturating_sub(rhs)),
                    (UInt(lhs), UInt(rhs), 4) => {
                        UInt((lhs as u32).saturating_sub(rhs as u32) as u64)
                    }
                    (UInt(lhs), UInt(rhs), 8) => UInt(lhs.saturating_sub(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Powf(op) => const_eval_float!(op.lhs, op.rhs; num_traits::Float::powf),
        // powf is fast enough for const eval
        Arithmetic::Powi(op) => {
            const_eval_float!(op.lhs, op.rhs; num_traits::Float::powf)
        }
        Arithmetic::ModFloor(op) => const_eval_int!(op.lhs, op.rhs; num_integer::mod_floor),
        Arithmetic::Rem(op) => const_eval!(% op.lhs, op.rhs),
        Arithmetic::MulHi(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => {
                        let mul = (lhs * rhs) >> 32;
                        ConstantValue::Int(mul as i32 as i64)
                    }
                    (UInt(lhs), UInt(rhs)) => {
                        let mul = (lhs * rhs) >> 32;
                        ConstantValue::UInt(mul as u32 as u64)
                    }
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Max(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs.max(rhs)),
                    (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs.max(rhs)),
                    (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs.max(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Min(op) => {
            use ConstantValue::*;
            let ty = op.lhs.ty;
            if let (Some(lhs), Some(rhs)) = (op.lhs.as_const(), op.rhs.as_const()) {
                let rhs = rhs.cast_to(ty);
                Some(match (lhs, rhs) {
                    (Int(lhs), Int(rhs)) => ConstantValue::Int(lhs.min(rhs)),
                    (Float(lhs), Float(rhs)) => ConstantValue::Float(lhs.min(rhs)),
                    (UInt(lhs), UInt(rhs)) => ConstantValue::UInt(lhs.min(rhs)),
                    _ => unreachable!(),
                })
            } else {
                None
            }
        }
        Arithmetic::Dot(op) => const_eval!(*op.lhs, op.rhs),

        Arithmetic::Abs(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.abs()),
                Float(input) => ConstantValue::Float(input.abs()),
                _ => unreachable!(),
            })
        }
        Arithmetic::Exp(op) => const_eval_float!(op.input; num_traits::Float::exp),
        Arithmetic::Log(op) => const_eval_float!(op.input; num_traits::Float::ln),
        Arithmetic::Log1p(op) => const_eval_float!(op.input; num_traits::Float::ln_1p),
        Arithmetic::Cos(op) => const_eval_float!(op.input; num_traits::Float::cos),
        Arithmetic::Sin(op) => const_eval_float!(op.input; num_traits::Float::sin),
        Arithmetic::Tan(op) => const_eval_float!(op.input; num_traits::Float::tan),
        Arithmetic::Tanh(op) => const_eval_float!(op.input; num_traits::Float::tanh),
        Arithmetic::Sinh(op) => const_eval_float!(op.input; num_traits::Float::sinh),
        Arithmetic::Cosh(op) => const_eval_float!(op.input; num_traits::Float::cosh),
        Arithmetic::ArcCos(op) => const_eval_float!(op.input; num_traits::Float::acos),
        Arithmetic::ArcSin(op) => const_eval_float!(op.input; num_traits::Float::asin),
        Arithmetic::ArcTan(op) => const_eval_float!(op.input; num_traits::Float::atan),
        Arithmetic::ArcSinh(op) => const_eval_float!(op.input; num_traits::Float::asinh),
        Arithmetic::ArcCosh(op) => const_eval_float!(op.input; num_traits::Float::acosh),
        Arithmetic::ArcTanh(op) => const_eval_float!(op.input; num_traits::Float::atanh),
        Arithmetic::Degrees(op) => const_eval_float!(op.input; num_traits::Float::to_degrees),
        Arithmetic::Radians(op) => const_eval_float!(op.input; num_traits::Float::to_radians),
        Arithmetic::ArcTan2(op) => const_eval_float!(op.lhs, op.rhs; num_traits::Float::atan2),
        Arithmetic::Sqrt(op) => const_eval_float!(op.input; num_traits::Float::sqrt),
        Arithmetic::Hypot(op) => const_eval_float!(op.lhs, op.rhs; num_traits::Float::hypot),
        Arithmetic::Rhypot(op) => {
            let hypot = const_eval_float!(op.lhs, op.rhs; num_traits::Float::hypot)?;
            let ConstantValue::Float(val) = hypot else {
                unreachable!()
            };
            Some(ConstantValue::Float(1.0 / val))
        }
        Arithmetic::InverseSqrt(op) => {
            let sqrt = const_eval_float!(op.input; num_traits::Float::sqrt)?;
            let ConstantValue::Float(val) = sqrt else {
                unreachable!()
            };
            Some(ConstantValue::Float(1.0 / val))
        }
        Arithmetic::Round(op) => const_eval_float!(op.input; num_traits::Float::round),
        Arithmetic::Floor(op) => const_eval_float!(op.input; num_traits::Float::floor),
        Arithmetic::Ceil(op) => const_eval_float!(op.input; num_traits::Float::ceil),
        Arithmetic::Trunc(op) => const_eval_float!(op.input; num_traits::Float::trunc),
        Arithmetic::Recip(op) => const_eval_float!(op.input; num_traits::Float::recip),
        Arithmetic::Neg(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(-input),
                Float(input) => ConstantValue::Float(-input),
                _ => unreachable!(),
            })
        }

        Arithmetic::Fma(op) => {
            use ConstantValue::*;
            let ty = op.a.ty;
            let a = op.a.as_const();
            let b = op.b.as_const();
            let c = op.c.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(ty);
                let c = c.cast_to(ty);
                match (a, b, c) {
                    (Float(a), Float(b), Float(c)) => ConstantValue::Float(a * b + c),
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Clamp(op) => {
            use ConstantValue::*;
            let ty = op.input.ty;
            let a = op.input.as_const();
            let b = op.min_value.as_const();
            let c = op.max_value.as_const();

            a.zip(b).zip(c).map(|((a, b), c)| {
                let b = b.cast_to(ty);
                let c = c.cast_to(ty);
                match (a, b, c) {
                    (Int(a), Int(b), Int(c)) => ConstantValue::Int(a.clamp(b, c)),
                    (Float(a), Float(b), Float(c)) => ConstantValue::Float(a.clamp(b, c)),
                    (UInt(a), UInt(b), UInt(c)) => ConstantValue::UInt(a.clamp(b, c)),
                    _ => unreachable!(),
                }
            })
        }
        Arithmetic::Erf(_)
        | Arithmetic::Magnitude(_)
        | Arithmetic::Normalize(_)
        | Arithmetic::VectorSum(_) => None,
    }
}

fn try_const_eval_cmp(op: &mut Comparison) -> Option<ConstantValue> {
    match op {
        Comparison::Equal(op) => const_eval_cmp!(== op.lhs, op.rhs),
        Comparison::NotEqual(op) => const_eval_cmp!(!= op.lhs, op.rhs),
        Comparison::Lower(op) => const_eval_cmp!(< op.lhs, op.rhs),
        Comparison::Greater(op) => const_eval_cmp!(> op.lhs, op.rhs),
        Comparison::LowerEqual(op) => const_eval_cmp!(<= op.lhs, op.rhs),
        Comparison::GreaterEqual(op) => const_eval_cmp!(>= op.lhs, op.rhs),
        Comparison::IsNan(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Float(val) => Bool(val.is_nan()),
                // Integers, bools, uints can't be NaN, so always false
                Int(_) | UInt(_) | Bool(_) => Bool(false),
            })
        }
        Comparison::IsInf(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Float(val) => Bool(val.is_infinite()),
                // Integers, bools, uints can't be infinite, so always false
                Int(_) | UInt(_) | Bool(_) => Bool(false),
            })
        }
    }
}

fn try_const_eval_bitwise(op: &mut Bitwise) -> Option<ConstantValue> {
    match op {
        Bitwise::BitwiseAnd(op) => const_eval_int!(&op.lhs, op.rhs),
        Bitwise::BitwiseOr(op) => const_eval_int!(| op.lhs, op.rhs),
        Bitwise::BitwiseXor(op) => const_eval_int!(^ op.lhs, op.rhs),
        Bitwise::ShiftLeft(op) => const_eval_int!(<< op.lhs, op.rhs),
        Bitwise::ShiftRight(op) => const_eval_int!(>> op.lhs, op.rhs),
        Bitwise::BitwiseNot(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(!input),
                UInt(input) => ConstantValue::UInt(!input),
                _ => unreachable!(),
            })
        }
        Bitwise::CountOnes(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.count_ones() as i64),
                UInt(input) => ConstantValue::UInt(input.count_ones() as u64),
                _ => unreachable!(),
            })
        }
        Bitwise::ReverseBits(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Int(input) => ConstantValue::Int(input.reverse_bits()),
                UInt(input) => ConstantValue::UInt(input.reverse_bits()),
                _ => unreachable!(),
            })
        }
        Bitwise::LeadingZeros(_) | Bitwise::TrailingZeros(_) | Bitwise::FindFirstSet(_) => {
            // Depends too much on type width and Rust semantics, leave this one out of const eval
            None
        }
    }
}

fn try_const_eval_operator(op: &mut Operator, out_ty: Option<Type>) -> Option<ConstantValue> {
    match op {
        Operator::And(op) => const_eval_bool!(&&op.lhs, op.rhs),
        Operator::Or(op) => const_eval_bool!(|| op.lhs, op.rhs),
        Operator::Not(op) => {
            use ConstantValue::*;
            op.input.as_const().map(|input| match input {
                Bool(input) => ConstantValue::Bool(!input),
                _ => unreachable!(),
            })
        }
        Operator::Cast(op) => op.input.as_const().map(|val| val.cast_to(out_ty.unwrap())),
        Operator::InitVector(_)
        | Operator::InsertComponent(_)
        | Operator::ExtractComponent(_)
        | Operator::Reinterpret(_)
        | Operator::Select(_) => None,
    }
}

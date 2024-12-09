use std::collections::HashMap;

use cubecl_core::ir::{
    BinaryOperator, ClampOperator, ConstantScalarValue, FmaOperator, Item, LineInitOperator,
    Metadata, Operation, Operator, Select, UnaryOperator, Variable, VariableKind,
};
use float_ord::FloatOrd;
use smallvec::SmallVec;

use super::{Constant, Expression, Local, OpId, Value};

impl Expression {
    pub fn to_operation(&self, leaders: &HashMap<u32, Value>) -> Operation {
        match self {
            Expression::Copy(val, _) => {
                let input = leaders[val].as_var();
                Operation::Copy(input)
            }
            Expression::Value(value) | Expression::Volatile(value) => {
                Operation::Copy(value.as_var())
            }
            Expression::Instruction(instruction) => {
                let args = instruction
                    .args
                    .iter()
                    .map(|val| leaders[val].as_var())
                    .collect::<SmallVec<[Variable; 4]>>();
                match instruction.op {
                    OpId::Add => Operator::Add(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Fma => Operator::Fma(FmaOperator {
                        a: args[0],
                        b: args[1],
                        c: args[2],
                    })
                    .into(),
                    OpId::Sub => Operator::Sub(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Mul => Operator::Mul(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Div => Operator::Div(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Abs => Operator::Abs(UnaryOperator { input: args[0] }).into(),
                    OpId::Exp => Operator::Exp(UnaryOperator { input: args[0] }).into(),
                    OpId::Log => Operator::Log(UnaryOperator { input: args[0] }).into(),
                    OpId::Log1p => Operator::Log1p(UnaryOperator { input: args[0] }).into(),
                    OpId::Cos => Operator::Cos(UnaryOperator { input: args[0] }).into(),
                    OpId::Sin => Operator::Sin(UnaryOperator { input: args[0] }).into(),
                    OpId::Tanh => Operator::Tanh(UnaryOperator { input: args[0] }).into(),
                    OpId::Powf => Operator::Powf(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Sqrt => Operator::Sqrt(UnaryOperator { input: args[0] }).into(),
                    OpId::Round => Operator::Round(UnaryOperator { input: args[0] }).into(),
                    OpId::Floor => Operator::Floor(UnaryOperator { input: args[0] }).into(),
                    OpId::Ceil => Operator::Ceil(UnaryOperator { input: args[0] }).into(),
                    OpId::Erf => Operator::Erf(UnaryOperator { input: args[0] }).into(),
                    OpId::Recip => Operator::Recip(UnaryOperator { input: args[0] }).into(),
                    OpId::Equal => Operator::Equal(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::NotEqual => Operator::NotEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Lower => Operator::Lower(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Clamp => Operator::Clamp(ClampOperator {
                        input: args[0],
                        min_value: args[1],
                        max_value: args[2],
                    })
                    .into(),
                    OpId::Greater => Operator::Greater(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::LowerEqual => Operator::LowerEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::GreaterEqual => Operator::GreaterEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Modulo => Operator::Modulo(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Index => Operator::Index(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::InitLine => Operator::InitLine(LineInitOperator {
                        inputs: args.into_vec(),
                    })
                    .into(),
                    OpId::And => Operator::And(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Or => Operator::Or(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Not => Operator::Not(UnaryOperator { input: args[0] }).into(),
                    OpId::Neg => Operator::Neg(UnaryOperator { input: args[0] }).into(),
                    OpId::Max => Operator::Max(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Min => Operator::Min(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::BitwiseAnd => Operator::BitwiseAnd(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::BitwiseOr => Operator::BitwiseOr(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::BitwiseXor => Operator::BitwiseXor(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::ShiftLeft => Operator::ShiftLeft(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::ShiftRight => Operator::ShiftRight(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Remainder => Operator::Remainder(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Magnitude => Operator::Magnitude(UnaryOperator { input: args[0] }).into(),
                    OpId::Normalize => Operator::Normalize(UnaryOperator { input: args[0] }).into(),
                    OpId::Dot => Operator::Dot(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                    })
                    .into(),
                    OpId::Select => Operator::Select(Select {
                        cond: args[0],
                        then: args[1],
                        or_else: args[2],
                    })
                    .into(),
                    OpId::Bitcast => Operator::Bitcast(UnaryOperator { input: args[0] }).into(),
                    OpId::Rank => Metadata::Rank { var: args[0] }.into(),
                    OpId::Length => Metadata::Length { var: args[0] }.into(),
                    OpId::BufferLength => Metadata::BufferLength { var: args[0] }.into(),
                    OpId::Shape => Metadata::Shape {
                        var: args[0],
                        dim: args[1],
                    }
                    .into(),
                    OpId::Stride => Metadata::Stride {
                        var: args[0],
                        dim: args[1],
                    }
                    .into(),
                    OpId::Cast => Operator::Cast(UnaryOperator { input: args[0] }).into(),
                }
            }
            Expression::Phi(_) => unreachable!("Phi can't be made into operation"),
        }
    }
}

impl Value {
    pub(crate) fn as_var(&self) -> Variable {
        match self {
            Value::Constant(val) => Variable::constant((*val).into()),
            Value::Local(Local {
                id,
                depth,
                version: 0,
                item,
            }) => Variable::new(
                VariableKind::LocalBinding {
                    id: *id,
                    depth: *depth,
                },
                *item,
            ),
            Value::Local(Local {
                id,
                depth,
                version,
                item,
            }) => Variable::new(
                VariableKind::Versioned {
                    id: *id,
                    depth: *depth,
                    version: *version,
                },
                *item,
            ),
            Value::Input(id, item) => Variable::new(VariableKind::GlobalInputArray(*id), *item),
            Value::Scalar(id, elem) => {
                Variable::new(VariableKind::GlobalScalar(*id), Item::new(*elem))
            }
            Value::ConstArray(id, item, len) => Variable::new(
                VariableKind::ConstantArray {
                    id: *id,
                    length: *len,
                },
                *item,
            ),
            Value::Builtin(builtin) => Variable::builtin(*builtin),
            Value::Output(id, item) => Variable::new(VariableKind::GlobalOutputArray(*id), *item),
            Value::Slice(id, depth, item) => Variable::new(
                VariableKind::Slice {
                    id: *id,
                    depth: *depth,
                },
                *item,
            ),
        }
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let item = var.item;
    let val = match var.kind {
        VariableKind::GlobalInputArray(id) => Value::Input(id, item),
        VariableKind::GlobalOutputArray(id) => Value::Output(id, item),
        VariableKind::GlobalScalar(id) => Value::Scalar(id, item.elem),
        VariableKind::Versioned { id, depth, version } => Value::Local(Local {
            id,
            depth,
            version,
            item,
        }),
        VariableKind::LocalBinding { id, depth } => Value::Local(Local {
            id,
            depth,
            version: 0,
            item,
        }),
        VariableKind::ConstantScalar(val) => Value::Constant(val.into()),
        VariableKind::ConstantArray { id, length } => Value::ConstArray(id, item, length),
        VariableKind::Local { .. }
        | VariableKind::SharedMemory { .. }
        | VariableKind::LocalArray { .. }
        | VariableKind::Matrix { .. } => None?,
        VariableKind::Slice { id, depth } => Value::Slice(id, depth, item),
        VariableKind::Builtin(builtin) => Value::Builtin(builtin),
        VariableKind::Ptr { id, depth } => todo!(),
    };
    Some(val)
}

pub fn id_of_op(op: &Operator) -> OpId {
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
        Operator::Cast(_) => OpId::Cast,
        Operator::Bitcast(_) => OpId::Bitcast,
        _ => unreachable!(),
    }
}

pub fn id_of_meta(meta: &Metadata) -> OpId {
    match meta {
        Metadata::Stride { .. } => OpId::Stride,
        Metadata::Shape { .. } => OpId::Shape,
        Metadata::Length { .. } => OpId::Length,
        Metadata::BufferLength { .. } => OpId::BufferLength,
        Metadata::Rank { .. } => OpId::Rank,
    }
}

impl From<Constant> for ConstantScalarValue {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(val, kind) => ConstantScalarValue::Int(val, kind),
            Constant::Float(val, kind) => ConstantScalarValue::Float(val.0, kind),
            Constant::UInt(val, kind) => ConstantScalarValue::UInt(val, kind),
            Constant::Bool(val) => ConstantScalarValue::Bool(val),
        }
    }
}

impl From<ConstantScalarValue> for Constant {
    fn from(value: ConstantScalarValue) -> Self {
        match value {
            ConstantScalarValue::Int(val, int_kind) => Constant::Int(val, int_kind),
            ConstantScalarValue::Float(val, float_kind) => {
                Constant::Float(FloatOrd(val), float_kind)
            }
            ConstantScalarValue::UInt(val, kind) => Constant::UInt(val, kind),
            ConstantScalarValue::Bool(val) => Constant::Bool(val),
        }
    }
}

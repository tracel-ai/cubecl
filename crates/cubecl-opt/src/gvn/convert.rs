use std::collections::HashMap;

use cubecl_core::ir::{
    BinaryOperator, Branch, ClampOperator, ConstantScalarValue, FmaOperator, LineInitOperator,
    Metadata, Operation, Operator, Select, UnaryOperator, Variable,
};
use float_ord::FloatOrd;
use smallvec::SmallVec;

use super::{Builtin, Constant, Expression, Local, OpId, Value};

impl Expression {
    pub fn to_operation(&self, leaders: &HashMap<u32, Value>, out: Variable) -> Operation {
        match self {
            Expression::Copy(val, _) => {
                let input = leaders[val].as_var();
                Operator::Assign(UnaryOperator { input, out }).into()
            }
            Expression::Value(value) | Expression::Volatile(value) => {
                Operator::Assign(UnaryOperator {
                    input: value.as_var(),
                    out,
                })
                .into()
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
                        out,
                    })
                    .into(),
                    OpId::Fma => Operator::Fma(FmaOperator {
                        a: args[0],
                        b: args[1],
                        c: args[2],
                        out,
                    })
                    .into(),
                    OpId::Sub => Operator::Sub(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Mul => Operator::Mul(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Div => Operator::Div(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Abs => Operator::Abs(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Exp => Operator::Exp(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Log => Operator::Log(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Log1p => Operator::Log1p(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Cos => Operator::Cos(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Sin => Operator::Sin(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Tanh => Operator::Tanh(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Powf => Operator::Powf(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Sqrt => Operator::Sqrt(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Round => Operator::Round(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Floor => Operator::Floor(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Ceil => Operator::Ceil(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Erf => Operator::Erf(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Recip => Operator::Recip(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Equal => Operator::Equal(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::NotEqual => Operator::NotEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Lower => Operator::Lower(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Clamp => Operator::Clamp(ClampOperator {
                        input: args[0],
                        min_value: args[1],
                        max_value: args[2],
                        out,
                    })
                    .into(),
                    OpId::Greater => Operator::Greater(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::LowerEqual => Operator::LowerEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::GreaterEqual => Operator::GreaterEqual(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Modulo => Operator::Modulo(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Index => Operator::Index(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::InitLine => Operator::InitLine(LineInitOperator {
                        inputs: args.into_vec(),
                        out,
                    })
                    .into(),
                    OpId::And => Operator::And(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Or => Operator::Or(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Not => Operator::Not(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Neg => Operator::Neg(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Max => Operator::Max(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Min => Operator::Min(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::BitwiseAnd => Operator::BitwiseAnd(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::BitwiseOr => Operator::BitwiseOr(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::BitwiseXor => Operator::BitwiseXor(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::ShiftLeft => Operator::ShiftLeft(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::ShiftRight => Operator::ShiftRight(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Remainder => Operator::Remainder(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Magnitude => Operator::Magnitude(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Normalize => Operator::Normalize(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Dot => Operator::Dot(BinaryOperator {
                        lhs: args[0],
                        rhs: args[1],
                        out,
                    })
                    .into(),
                    OpId::Select => Branch::Select(Select {
                        cond: args[0],
                        then: args[1],
                        or_else: args[2],
                        out,
                    })
                    .into(),
                    OpId::Bitcast => Operator::Bitcast(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                    OpId::Length => Metadata::Length { var: args[0], out }.into(),
                    OpId::Shape => Metadata::Shape {
                        var: args[0],
                        dim: args[1],
                        out,
                    }
                    .into(),
                    OpId::Stride => Metadata::Stride {
                        var: args[0],
                        dim: args[1],
                        out,
                    }
                    .into(),
                    OpId::Cast => Operator::Assign(UnaryOperator {
                        input: args[0],
                        out,
                    })
                    .into(),
                }
            }
            Expression::Phi(_) => todo!("Phi can't be made into operation"),
        }
    }
}

impl Value {
    pub(crate) fn as_var(&self) -> Variable {
        match self {
            Value::Constant(val) => Variable::ConstantScalar(match val {
                Constant::Int(val, kind) => ConstantScalarValue::Int(*val, *kind),
                Constant::Float(val, kind) => ConstantScalarValue::Float(val.0, *kind),
                Constant::UInt(val) => ConstantScalarValue::UInt(*val),
                Constant::Bool(val) => ConstantScalarValue::Bool(*val),
            }),
            Value::Local(Local(id, depth, 0, item)) => Variable::LocalBinding {
                id: *id,
                item: *item,
                depth: *depth,
            },
            Value::Local(Local(id, depth, v, item)) => Variable::Versioned {
                id: *id,
                item: *item,
                depth: *depth,
                version: *v,
            },
            Value::Input(id, item) => Variable::GlobalInputArray {
                id: *id,
                item: *item,
            },
            Value::Scalar(id, elem) => Variable::GlobalScalar {
                id: *id,
                elem: *elem,
            },
            Value::ConstArray(id, item, len) => Variable::ConstantArray {
                id: *id,
                item: *item,
                length: *len,
            },
            Value::Builtin(builtin) => builtin.as_var(),
            Value::Output(id, item) => Variable::GlobalOutputArray {
                id: *id,
                item: *item,
            },
            Value::Slice(id, depth, item) => Variable::Slice {
                id: *id,
                item: *item,
                depth: *depth,
            },
        }
    }
}

impl Builtin {
    pub fn as_var(&self) -> Variable {
        match self {
            Builtin::Rank => Variable::Rank,
            Builtin::UnitPos => Variable::UnitPos,
            Builtin::UnitPosX => Variable::UnitPosX,
            Builtin::UnitPosY => Variable::UnitPosY,
            Builtin::UnitPosZ => Variable::UnitPosZ,
            Builtin::CubePos => Variable::CubePos,
            Builtin::CubePosX => Variable::CubePosX,
            Builtin::CubePosY => Variable::CubePosY,
            Builtin::CubePosZ => Variable::CubePosZ,
            Builtin::CubeDim => Variable::CubeDim,
            Builtin::CubeDimX => Variable::CubeDimX,
            Builtin::CubeDimY => Variable::CubeDimY,
            Builtin::CubeDimZ => Variable::CubeDimZ,
            Builtin::CubeCount => Variable::CubeCount,
            Builtin::CubeCountX => Variable::CubeCountX,
            Builtin::CubeCountY => Variable::CubeCountY,
            Builtin::CubeCountZ => Variable::CubeCountZ,
            Builtin::SubcubeDim => Variable::SubcubeDim,
            Builtin::AbsolutePos => Variable::AbsolutePos,
            Builtin::AbsolutePosX => Variable::AbsolutePosX,
            Builtin::AbsolutePosY => Variable::AbsolutePosY,
            Builtin::AbsolutePosZ => Variable::AbsolutePosZ,
        }
    }
}

pub fn value_of_var(var: &Variable) -> Option<Value> {
    let val = match var {
        Variable::GlobalInputArray { id, item } => Value::Input(*id, *item),
        Variable::GlobalScalar { id, elem } => Value::Scalar(*id, *elem),
        Variable::GlobalOutputArray { id, item } => Value::Output(*id, *item),
        Variable::Versioned {
            id,
            depth,
            version,
            item,
        } => Value::Local(Local(*id, *depth, *version, *item)),
        Variable::LocalBinding { id, depth, item } => Value::Local(Local(*id, *depth, 0, *item)),
        Variable::ConstantScalar(val) => Value::Constant((*val).into()),
        Variable::ConstantArray { id, item, length } => Value::ConstArray(*id, *item, *length),
        Variable::Local { .. }
        | Variable::SharedMemory { .. }
        | Variable::LocalArray { .. }
        | Variable::Matrix { .. } => None?,
        Variable::Slice { id, depth, item } => Value::Slice(*id, *depth, *item),
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
        Operator::Bitcast(_) => OpId::Bitcast,
        _ => unreachable!(),
    }
}

pub fn id_of_meta(meta: &Metadata) -> OpId {
    match meta {
        Metadata::Stride { .. } => OpId::Stride,
        Metadata::Shape { .. } => OpId::Shape,
        Metadata::Length { .. } => OpId::Length,
    }
}

impl From<Constant> for ConstantScalarValue {
    fn from(value: Constant) -> Self {
        match value {
            Constant::Int(val, kind) => ConstantScalarValue::Int(val, kind),
            Constant::Float(val, kind) => ConstantScalarValue::Float(val.0, kind),
            Constant::UInt(val) => ConstantScalarValue::UInt(val),
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
            ConstantScalarValue::UInt(val) => Constant::UInt(val),
            ConstantScalarValue::Bool(val) => Constant::Bool(val),
        }
    }
}

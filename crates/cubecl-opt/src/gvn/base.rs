use std::collections::HashMap;

use cubecl_core::ir::{ConstantScalarValue, Elem, FloatKind, IntKind};
use float_ord::FloatOrd;
use smallvec::SmallVec;

use crate::PhiInstruction;

#[derive(Debug)]
pub struct ValueTable {
    pub(crate) value_numbers: HashMap<Value, u32>,
    pub(crate) expression_numbers: HashMap<Expression, u32>,
    pub(crate) phi_numbers: HashMap<u32, PhiInstruction>,

    pub(crate) next_expr_num: u32,
    pub(crate) next_value_num: u32,

    pub(crate) expressions: HashMap<u32, Expression>,
}

impl Default for ValueTable {
    fn default() -> Self {
        Self {
            value_numbers: Default::default(),
            expression_numbers: Default::default(),
            phi_numbers: Default::default(),
            next_expr_num: 0,
            next_value_num: 1,
            expressions: Default::default(),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Local(pub u16, pub u8, pub u16);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Constant {
    Int(i64, IntKind),
    Float(FloatOrd<f64>, FloatKind),
    UInt(u64),
    Bool(bool),
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Value {
    Constant(Constant),
    Local(Local),
    Input(u16),
    Scalar(u16, Elem),
    ConstArray(u16),
    Builtin(Builtin),
    // Metadata only
    Output(u16),
    Slice(u16, u8),
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub enum Expression {
    Instruction(Instruction),
    Copy(u32),
    Value(Value),
    Volatile(Value),
}

impl Expression {
    pub fn depends_on(&self) -> SmallVec<[u32; 4]> {
        match self {
            Expression::Instruction(instruction) => instruction.args.clone(),
            Expression::Copy(val) => SmallVec::from_slice(&[*val]),
            Expression::Volatile(_) | Expression::Value(_) => SmallVec::new(),
        }
    }
}

impl From<Instruction> for Expression {
    fn from(value: Instruction) -> Self {
        Expression::Instruction(value)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub struct Instruction {
    op: OpId,
    commutative: bool,
    args: SmallVec<[u32; 4]>,
}

impl Instruction {
    pub fn new(op: OpId, args: &[u32]) -> Self {
        Self {
            op,
            commutative: false,
            args: SmallVec::from_slice(args),
        }
    }

    pub fn commutative(op: OpId, args: &[u32]) -> Self {
        Self {
            op,
            commutative: true,
            args: SmallVec::from_slice(args),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum OpId {
    Add,
    Fma,
    Sub,
    Mul,
    Div,
    Abs,
    Exp,
    Log,
    Log1p,
    Cos,
    Sin,
    Tanh,
    Powf,
    Sqrt,
    Round,
    Floor,
    Ceil,
    Erf,
    Recip,
    Equal,
    NotEqual,
    Lower,
    Clamp,
    Greater,
    LowerEqual,
    GreaterEqual,
    Modulo,
    Index,
    InitLine,
    And,
    Or,
    Not,
    Neg,
    Max,
    Min,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    Remainder,
    Magnitude,
    Normalize,
    Dot,
    Select,
    Bitcast,
    Length,
    Shape,
    Stride,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Builtin {
    Rank,
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    SubcubeDim,
    AbsolutePos,
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
}

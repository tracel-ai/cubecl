use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use cubecl_core::ir::{
    ConstantScalarValue, Elem, FloatKind, IntKind, Metadata, Operation, Operator, Variable,
};
use float_ord::FloatOrd;

use crate::AtomicCounter;

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

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub enum GlobalValue {
    Constant(Constant),
    Variable(u16, u8, u16),
    Input(u16),
    Scalar(u16, Elem),
    ConstArray(u16),
    Operator { op: OpId, operands: Vec<usize> },
    Builtin(Builtin),
    // Metadata only
    Output(u16),
    Slice(u16, u8),
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

    Length,
    Shape,
    Stride,

    Assign,
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

#[derive(Default, Clone, Debug)]
pub struct GlobalNumberTable {
    pub(crate) classes: HashMap<GlobalValue, usize>,
    pub(crate) leaders: HashMap<usize, Variable>,
    pub(crate) global_classes: Rc<RefCell<HashMap<GlobalValue, usize>>>,
    class_id: AtomicCounter,
}

impl PartialEq for GlobalNumberTable {
    fn eq(&self, other: &Self) -> bool {
        self.classes == other.classes && self.leaders == other.leaders
    }
}

pub struct Instruction {
    out: GlobalValue,
    rhs: GlobalValue,
    class: usize,
}

impl GlobalNumberTable {
    pub fn new(class_id: AtomicCounter, globals: Rc<RefCell<HashMap<GlobalValue, usize>>>) -> Self {
        Self {
            class_id,
            global_classes: globals,
            ..Default::default()
        }
    }

    pub(crate) fn intersection(self, other: GlobalNumberTable) -> GlobalNumberTable {
        let this = self.classes.values().copied().collect::<HashSet<_>>();
        let avail_other = other.classes.values().copied().collect::<HashSet<_>>();
        let intersect = this.intersection(&avail_other).collect::<HashSet<_>>();
        let mut classes = self
            .classes
            .into_iter()
            .chain(other.classes)
            .filter(|(_, v)| intersect.contains(v))
            .collect::<HashMap<_, _>>();
        classes.extend(self.global_classes.borrow().clone());
        let leaders = self
            .leaders
            .into_iter()
            .chain(other.leaders)
            .filter(|(k, _)| intersect.contains(k))
            .collect::<HashMap<_, _>>();
        Self {
            classes,
            leaders,
            class_id: self.class_id,
            global_classes: self.global_classes,
        }
    }

    fn class_of(&mut self, value: GlobalValue, out: GlobalValue) -> usize {
        if let Some(existing) = self.classes.get(&value).copied() {
            self.classes.insert(out, existing);
            existing
        } else {
            let class = self.class_id.inc();

            self.classes.insert(value, class);
            self.classes.insert(out, class);

            class
        }
    }

    fn insert_var(&mut self, value: GlobalValue, class: usize) {
        self.classes.insert(value, class);
    }

    fn global_var(&mut self, value: GlobalValue) -> usize {
        let existing = { self.global_classes.borrow().get(&value).copied() };
        if let Some(existing) = existing {
            self.classes.insert(value, existing);
            existing
        } else {
            let class = self.class_id.inc();
            self.global_classes
                .borrow_mut()
                .insert(value.clone(), class);
            self.classes.insert(value, class);
            class
        }
    }

    pub(crate) fn class_of_var(&mut self, var: &Variable) -> Option<usize> {
        let val = match var {
            // May be mutable, not safe to number
            Variable::GlobalOutputArray { .. }
            | Variable::Matrix { .. }
            | Variable::Local { .. }
            | Variable::SharedMemory { .. }
            | Variable::LocalArray { .. }
            | Variable::Slice { .. } => None?,
            Variable::GlobalInputArray { id, .. } => self.global_var(GlobalValue::Input(*id)),
            Variable::GlobalScalar { id, elem } => self.global_var(GlobalValue::Scalar(*id, *elem)),
            Variable::ConstantArray { id, .. } => self.global_var(GlobalValue::ConstArray(*id)),
            Variable::Versioned {
                id, depth, version, ..
            } => *self
                .classes
                .get(&GlobalValue::Variable(*id, *depth, *version))?,
            Variable::LocalBinding { id, depth, .. } => {
                *self.classes.get(&GlobalValue::Variable(*id, *depth, 0))?
            }
            Variable::ConstantScalar(val) => self.global_var(GlobalValue::Constant((*val).into())),
            Variable::Rank => self.global_var(GlobalValue::Builtin(Builtin::Rank)),
            Variable::UnitPos => self.global_var(GlobalValue::Builtin(Builtin::UnitPos)),
            Variable::UnitPosX => self.global_var(GlobalValue::Builtin(Builtin::UnitPosX)),
            Variable::UnitPosY => self.global_var(GlobalValue::Builtin(Builtin::UnitPosY)),
            Variable::UnitPosZ => self.global_var(GlobalValue::Builtin(Builtin::UnitPosZ)),
            Variable::CubePos => self.global_var(GlobalValue::Builtin(Builtin::CubePos)),
            Variable::CubePosX => self.global_var(GlobalValue::Builtin(Builtin::CubePosX)),
            Variable::CubePosY => self.global_var(GlobalValue::Builtin(Builtin::CubePosY)),
            Variable::CubePosZ => self.global_var(GlobalValue::Builtin(Builtin::CubePosZ)),
            Variable::CubeDim => self.global_var(GlobalValue::Builtin(Builtin::CubeDim)),
            Variable::CubeDimX => self.global_var(GlobalValue::Builtin(Builtin::CubeDimX)),
            Variable::CubeDimY => self.global_var(GlobalValue::Builtin(Builtin::CubeDimY)),
            Variable::CubeDimZ => self.global_var(GlobalValue::Builtin(Builtin::CubeDimZ)),
            Variable::CubeCount => self.global_var(GlobalValue::Builtin(Builtin::CubeCount)),
            Variable::CubeCountX => self.global_var(GlobalValue::Builtin(Builtin::CubeCountX)),
            Variable::CubeCountY => self.global_var(GlobalValue::Builtin(Builtin::CubeCountY)),
            Variable::CubeCountZ => self.global_var(GlobalValue::Builtin(Builtin::CubeCountZ)),
            Variable::SubcubeDim => self.global_var(GlobalValue::Builtin(Builtin::SubcubeDim)),
            Variable::AbsolutePos => self.global_var(GlobalValue::Builtin(Builtin::AbsolutePos)),
            Variable::AbsolutePosX => self.global_var(GlobalValue::Builtin(Builtin::AbsolutePosX)),
            Variable::AbsolutePosY => self.global_var(GlobalValue::Builtin(Builtin::AbsolutePosY)),
            Variable::AbsolutePosZ => self.global_var(GlobalValue::Builtin(Builtin::AbsolutePosZ)),
        };
        Some(val)
    }

    pub(crate) fn instruction_of_operator(&mut self, operator: &Operator) -> Option<Instruction> {
        match operator {
            Operator::Assign(op) => {
                let input = self.class_of_var(&op.input)?;
                let out = value_of_var(&op.out)?;
                self.classes.insert(out.clone(), input);
                Some(Instruction {
                    out,
                    rhs: GlobalValue::Operator {
                        op: OpId::Assign,
                        operands: vec![input],
                    },
                    class: input,
                })
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
                let mut operands = vec![self.class_of_var(&op.lhs)?, self.class_of_var(&op.rhs)?];
                operands.sort();
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
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
                let operands = vec![self.class_of_var(&op.lhs)?, self.class_of_var(&op.rhs)?];
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }

            // Compare ops
            Operator::Lower(op)
            | Operator::Greater(op)
            | Operator::LowerEqual(op)
            | Operator::GreaterEqual(op) => {
                let lhs = self.class_of_var(&op.lhs)?;
                let rhs = self.class_of_var(&op.rhs)?;
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let inverse = cmp_inverse(&op);
                let value = GlobalValue::Operator {
                    op,
                    operands: vec![lhs, rhs],
                };
                let inv_value = GlobalValue::Operator {
                    op: inverse,
                    operands: vec![rhs, lhs],
                };
                let class = self.class_of(value.clone(), out.clone());
                self.classes.insert(inv_value, class);
                Some(Instruction {
                    out,
                    rhs: value,
                    class,
                })
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
                let operands = vec![self.class_of_var(&op.input)?];
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }

            Operator::Fma(op) => {
                let a = self.class_of_var(&op.a)?;
                let b = self.class_of_var(&op.b)?;
                let c = self.class_of_var(&op.c)?;
                let operands = vec![a, b, c];
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                let operands = vec![b, a, c];
                let value_2 = GlobalValue::Operator { op, operands };
                self.classes.insert(value_2, class);
                Some(Instruction { out, rhs, class })
            }
            Operator::Clamp(op) => {
                let val = self.class_of_var(&op.input)?;
                let min = self.class_of_var(&op.min_value)?;
                let max = self.class_of_var(&op.max_value)?;
                let out = value_of_var(&op.out)?;
                let operands = vec![val, min, max];
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }
            Operator::InitLine(op) => {
                let operands = op.inputs.iter().map(|it| self.class_of_var(it));
                let operands = operands.collect::<Option<_>>()?;
                let out = value_of_var(&op.out)?;
                let op = id_of_op(operator)?;
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }
            _ => None,
        }
    }

    fn class_of_metadata(&mut self, metadata: &Metadata) -> Option<Instruction> {
        let op = id_of_meta(metadata);
        match metadata {
            Metadata::Stride { dim, var, out } | Metadata::Shape { dim, var, out } => {
                let val = match var {
                    Variable::GlobalInputArray { id, .. } => GlobalValue::Input(*id),
                    Variable::GlobalOutputArray { id, .. } => GlobalValue::Output(*id),
                    _ => unreachable!("Stride/shape only exist for tensors"),
                };
                let var = self.global_var(val);
                let out = value_of_var(out)?;
                let operands = vec![var, self.class_of_var(dim)?];
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }
            Metadata::Length { var, out } => {
                let out = value_of_var(out)?;
                let var = match var {
                    Variable::GlobalInputArray { id, .. } => {
                        self.global_var(GlobalValue::Input(*id))
                    }
                    Variable::GlobalOutputArray { id, .. } => {
                        self.global_var(GlobalValue::Output(*id))
                    }
                    Variable::GlobalScalar { id, elem } => {
                        self.global_var(GlobalValue::Scalar(*id, *elem))
                    }
                    Variable::Slice { id, depth, .. } => *self
                        .classes
                        .entry(GlobalValue::Slice(*id, *depth))
                        .or_insert_with(|| self.class_id.inc()),
                    Variable::ConstantArray { length, .. }
                    | Variable::SharedMemory { length, .. }
                    | Variable::LocalArray { length, .. } => {
                        let constant = GlobalValue::Constant(Constant::UInt(*length as u64));
                        let class = self.global_var(constant);
                        self.classes.insert(out.clone(), class);
                        return Some(Instruction {
                            out,
                            rhs: GlobalValue::Operator {
                                op: OpId::Assign,
                                operands: vec![class],
                            },
                            class,
                        });
                    }
                    _ => unreachable!("Length only available on array"),
                };
                let operands = vec![var];
                let rhs = GlobalValue::Operator { op, operands };
                let class = self.class_of(rhs.clone(), out.clone());
                Some(Instruction { out, rhs, class })
            }
        }
    }

    pub(crate) fn class_of_operation(&mut self, operation: &Operation) -> Option<Instruction> {
        match operation {
            Operation::Operator(operator) => self.instruction_of_operator(operator),
            Operation::Metadata(metadata) => self.class_of_metadata(metadata),
            _ => None,
        }
    }
}

fn value_of_var(var: &Variable) -> Option<GlobalValue> {
    match var {
        Variable::Versioned {
            id, depth, version, ..
        } => Some(GlobalValue::Variable(*id, *depth, *version)),
        Variable::LocalBinding { id, depth, .. } => Some(GlobalValue::Variable(*id, *depth, 0)),
        _ => None,
    }
}

fn id_of_op(op: &Operator) -> Option<OpId> {
    let val = match op {
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
        _ => None?,
    };
    Some(val)
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

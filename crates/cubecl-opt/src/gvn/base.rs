use std::collections::HashMap;

use cubecl_core::ir::{Elem, FloatKind, IntKind};
use float_ord::FloatOrd;
use petgraph::{
    algo::dominators::{self, Dominators},
    graph::NodeIndex,
};
use smallvec::SmallVec;

use crate::{AtomicCounter, BasicBlock, Optimizer, PhiInstruction};

type PhiTranslateMap = HashMap<(u32, BasicBlock), u32>;

pub struct ValueTable {
    value_numbers: HashMap<Value, u32>,
    expression_numbers: HashMap<Expression, u32>,
    phi_numbers: HashMap<u32, PhiInstruction>,

    expr_id: AtomicCounter,
    value_id: AtomicCounter,

    expressions: HashMap<u32, Expression>,
    phi_translate_table: PhiTranslateMap,

    dominator_tree: Dominators<NodeIndex>,
    post_dom_tree: Dominators<NodeIndex>,
}

impl ValueTable {
    pub fn new(opt: &Optimizer) -> Self {
        let dominator_tree = dominators::simple_fast(&opt.program.graph, opt.entry());
        let mut rev_graph = opt.program.graph.clone();
        rev_graph.reverse();
        let post_dom_tree = dominators::simple_fast(&rev_graph, opt.ret);
        Self {
            value_numbers: Default::default(),
            expression_numbers: Default::default(),
            phi_numbers: Default::default(),
            expr_id: AtomicCounter::new(0),
            value_id: AtomicCounter::new(1),
            expressions: Default::default(),
            phi_translate_table: Default::default(),
            dominator_tree,
            post_dom_tree,
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
pub struct Expression {
    op: OpId,
    commutative: bool,
    args: SmallVec<[u32; 4]>,
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

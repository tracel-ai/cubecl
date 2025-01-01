use std::collections::{HashMap, HashSet};

use cubecl_core::{
    ir::{Builtin, ConstantScalarValue, Elem, FloatKind, IntKind, Item, UIntKind},
    prelude::CubePrimitive,
};
use float_ord::FloatOrd;
use petgraph::{
    algo::dominators::{self, Dominators},
    graph::NodeIndex,
    visit::{DfsPostOrder, Walker as _},
};
use smallvec::SmallVec;

use crate::{passes::OptimizerPass, AtomicCounter, Optimizer, PhiInstruction};

use super::{convert::value_of_var, BlockSets};

impl OptimizerPass for GvnPass {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        self.run(opt, &changes);
    }
}

#[derive(Debug, Clone)]
pub struct GvnPass {
    pub values: ValueTable,
    pub block_sets: HashMap<NodeIndex, BlockSets>,
    pub dominators: Dominators<NodeIndex>,
    pub post_doms: Dominators<NodeIndex>,
}

impl GvnPass {
    /// Run the GVN-PRE algorithm
    /// 1. Build forward and backward dominator trees
    /// 2. Run `build_sets` step to annotate the tree with value information
    /// 3. Insert expressions where they're needed to make partially redundant expressions fully
    ///     redundant
    /// 4. Replace fully redundant expressions with simple assignments from the leader of that
    ///     expression to `out`
    pub fn run(&mut self, opt: &mut Optimizer, changes: &AtomicCounter) {
        self.build_dominators(opt);
        self.build_sets(opt);
        self.insert(opt, changes);
        self.eliminate(opt, changes);
    }

    fn build_dominators(&mut self, opt: &mut Optimizer) {
        let post_order = DfsPostOrder::new(&opt.program.graph, opt.entry())
            .iter(&opt.program.graph)
            .collect::<Vec<_>>();
        for node in opt.node_ids() {
            if !post_order.contains(&node) {
                opt.program.remove_node(node);
            }
        }

        self.dominators = dominators::simple_fast(&opt.program.graph, opt.entry());
        let mut rev_graph = opt.program.graph.clone();
        rev_graph.reverse();
        self.post_doms = dominators::simple_fast(&rev_graph, opt.ret);
    }
}

/// A global value table that maps expressions and locals to the values they represent.
#[derive(Debug, Clone)]
pub struct ValueTable {
    pub(crate) value_numbers: HashMap<Value, u32>,
    pub(crate) expression_numbers: HashMap<Expression, u32>,

    pub(crate) next_expr_num: u32,
    pub(crate) next_value_num: u32,
}
impl ValueTable {
    pub(crate) fn insert_phi(&mut self, phi: &PhiInstruction, val: u32) {
        let expr = Expression::Phi(
            phi.entries
                .iter()
                .map(|it| (value_of_var(&it.value).unwrap(), it.block))
                .collect(),
        );
        let out = value_of_var(&phi.out).unwrap();
        self.expression_numbers.insert(expr, val);
        self.value_numbers.insert(out, val);
    }
}

impl Default for ValueTable {
    fn default() -> Self {
        Self {
            value_numbers: Default::default(),
            expression_numbers: Default::default(),
            next_expr_num: 0,
            next_value_num: 1,
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Local {
    pub id: u16,
    pub depth: u8,
    pub version: u16,
    pub item: Item,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Constant {
    Int(i64, IntKind),
    Float(FloatOrd<f64>, FloatKind),
    UInt(u64, UIntKind),
    Bool(bool),
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Value {
    Constant(Constant),
    Local(Local),
    Input(u16, Item),
    Scalar(u16, Elem),
    ConstArray(u16, Item, u32),
    Builtin(Builtin),
    // Metadata only
    Output(u16, Item),
    Slice(u16, u8, Item),
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub enum Expression {
    Instruction(Instruction),
    Copy(u32, Item),
    Value(Value),
    Volatile(Value),
    Phi(Vec<(Value, NodeIndex)>),
}

impl Expression {
    pub fn depends_on(&self) -> SmallVec<[u32; 4]> {
        match self {
            Expression::Instruction(instruction) => instruction.args.clone(),
            Expression::Copy(val, _) => SmallVec::from_slice(&[*val]),
            Expression::Phi(_) | Expression::Volatile(_) | Expression::Value(_) => SmallVec::new(),
        }
    }

    /// Whether the expression is a trivial copy (which does not need to be hoisted since it's free)
    pub fn is_simple(&self) -> bool {
        matches!(self, Expression::Copy(_, _))
    }

    pub fn item(&self) -> Item {
        match self {
            Expression::Instruction(instruction) => instruction.item,
            Expression::Copy(_, item) => *item,
            Expression::Value(value) => value.item(),
            Expression::Volatile(value) => value.item(),
            Expression::Phi(entries) => entries[0].0.item(),
        }
    }
}

impl Value {
    pub fn item(&self) -> Item {
        match self {
            Value::Constant(constant) => constant.item(),
            Value::Local(local) => local.item,
            Value::Input(_, item) => *item,
            Value::Scalar(_, elem) => Item::new(*elem),
            Value::ConstArray(_, item, _) => *item,
            Value::Builtin(_) => Item::new(u32::as_elem()),
            Value::Output(_, item) => *item,
            Value::Slice(_, _, item) => *item,
        }
    }
}

impl Constant {
    pub fn item(&self) -> Item {
        let val: ConstantScalarValue = (*self).into();
        Item::new(val.elem())
    }
}

impl From<Instruction> for Expression {
    fn from(value: Instruction) -> Self {
        Expression::Instruction(value)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub struct Instruction {
    pub(crate) op: OpId,
    pub(crate) commutative: bool,
    pub(crate) args: SmallVec<[u32; 4]>,
    pub(crate) item: Item,
}

impl Instruction {
    pub fn new(op: OpId, args: &[u32], item: Item) -> Self {
        Self {
            op,
            commutative: false,
            args: SmallVec::from_slice(args),
            item,
        }
    }

    pub fn commutative(op: OpId, args: &[u32], item: Item) -> Self {
        Self {
            op,
            commutative: true,
            args: SmallVec::from_slice(args),
            item,
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
    Rank,
    Length,
    BufferLength,
    Shape,
    Stride,
    Cast,
}

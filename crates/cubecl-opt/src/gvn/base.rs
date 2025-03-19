use std::collections::HashMap;

use cubecl_ir::{
    Builtin, ConstantScalarValue, Elem, FloatKind, Id, IntKind, Item, OpCode, UIntKind,
};
use float_ord::FloatOrd;
use petgraph::graph::NodeIndex;
use smallvec::SmallVec;

use crate::{AtomicCounter, Optimizer, PhiInstruction, passes::OptimizerPass};

use super::{GlobalValues, convert::value_of_var};

#[derive(Debug, Clone, Default)]
pub struct GvnPass;

impl OptimizerPass for GvnPass {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        self.run(opt, &changes);
    }
}

impl GvnPass {
    /// Run the GVN-PRE algorithm
    /// 1. Build forward and backward dominator trees
    /// 2. Run `build_sets` step to annotate the tree with value information
    /// 3. Insert expressions where they're needed to make partially redundant expressions fully
    ///    redundant
    /// 4. Replace fully redundant expressions with simple assignments from the leader of that
    ///    expression to `out`
    pub fn run(&mut self, opt: &mut Optimizer, changes: &AtomicCounter) {
        let analysis = opt.analysis::<GlobalValues>();

        analysis.0.borrow_mut().insert(opt, changes);
        analysis.0.borrow_mut().eliminate(opt, changes);
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
    pub id: Id,
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
    Input(Id, Item),
    Scalar(Id, Elem),
    ConstArray(Id, Item, u32),
    Builtin(Builtin),
    // Metadata only
    Output(Id, Item),
    Slice(Id, Item),
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
            Value::Builtin(_) => Item::new(Elem::UInt(UIntKind::U32)),
            Value::Output(_, item) => *item,
            Value::Slice(_, item) => *item,
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
    pub(crate) op: OpCode,
    pub(crate) commutative: bool,
    pub(crate) args: SmallVec<[u32; 4]>,
    pub(crate) item: Item,
}

impl Instruction {
    pub fn new(op: impl Into<OpCode>, args: &[u32], item: Item) -> Self {
        Self {
            op: op.into(),
            commutative: false,
            args: SmallVec::from_slice(args),
            item,
        }
    }

    pub fn commutative(op: impl Into<OpCode>, args: &[u32], item: Item) -> Self {
        Self {
            op: op.into(),
            commutative: true,
            args: SmallVec::from_slice(args),
            item,
        }
    }
}

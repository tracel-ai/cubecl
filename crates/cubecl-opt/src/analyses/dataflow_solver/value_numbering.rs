use core::{
    cell::{Cell, RefCell},
    fmt,
};

use alloc::{rc::Rc, vec::Vec};
use cubecl_ir::{
    interfaces::{
        Expression, ExpressionCanonicalize, ExpressionValue, side_effects::MemoryEffectsOp,
    },
    prelude::*,
};
use derive_more::{Eq, From, PartialEq};
use derive_new::new;
use pliron::{
    opts::dce::SideEffects,
    printable::{self, Printable},
    utils::table::HMap,
};

use crate::analyses::{
    alias_analysis::default_stack,
    dataflow_solver::{
        DataflowAnalysis, ProgramPoint,
        dead_code::is_block_live,
        sparse::{LatticeValue, SparseForward, SparseForwardDataflowAnalysis, SparseLattice},
    },
    memory_ssa::MemorySSA,
};

use super::{DataflowSolver, ReadRef, WriteRef};

#[derive(new, Clone, PartialEq, Eq, Hash)]
enum NumberingKey {
    Opaque(Value),
    Expression(Expression),
}

#[derive(new, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueNumber {
    value: u64,
}

impl Default for ValueNumber {
    fn default() -> Self {
        Self::UNINITIALIZED
    }
}

impl LatticeValue for ValueNumber {
    fn join(_this: &SparseLattice<Self>, _rhs: &Self) -> Self {
        panic!("Should not be called, only used for convenience so we can use `SparseLattice`")
    }
}

impl Printable for ValueNumber {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.is_initialized() {
            false => write!(f, "ValueNumber(Uninitialized)"),
            true => write!(f, "ValueNumber({})", self.value),
        }
    }
}

impl ValueNumber {
    pub const UNINITIALIZED: ValueNumber = ValueNumber { value: u64::MAX };

    pub fn value(&self) -> Option<u64> {
        self.is_initialized().then_some(self.value)
    }

    pub fn is_initialized(&self) -> bool {
        self != &Self::UNINITIALIZED
    }
}

#[derive(Clone, PartialEq, Eq, Default, From, Debug)]
pub enum ValueClass {
    #[default]
    Uninitialized,
    #[from]
    Expression(Expression),
    Opaque(Value),
}

impl Printable for ValueClass {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            ValueClass::Uninitialized => f.write_str("ValueClass::Uninitialized"),
            ValueClass::Expression(expr) => expr.fmt(ctx, state, f),
            ValueClass::Opaque(value) => write!(f, "ValueClass::Opaque({})", value.disp(ctx)),
        }
    }
}

impl LatticeValue for ValueClass {
    fn join(this: &SparseLattice<Self>, rhs: &Self) -> Self {
        if this.value() == rhs {
            return rhs.clone();
        }
        match (this.value(), rhs) {
            (ValueClass::Uninitialized, other) | (other, ValueClass::Uninitialized) => {
                other.clone()
            }
            _ => ValueClass::Opaque(this.anchor()),
        }
    }
}

type ValueClassLattice = SparseLattice<ValueClass>;
pub type ValueClassAnalysis = SparseForward<ValueClasses>;
pub type ValueNumberLattice = SparseLattice<ValueNumber>;

#[derive(Default)]
pub struct ValueClasses;

impl ValueClasses {
    fn get_value_number(&self, solver: &DataflowSolver, value: Value) -> Option<u64> {
        let number = solver.get_or_create::<ValueNumberLattice>(value);
        number.deref().use_def_subscribe::<SparseForward<Self>>();
        number.deref().value().value()
    }
}

impl SparseForwardDataflowAnalysis for ValueClasses {
    type LatticeValue = ValueClass;

    fn verify(solver: &DataflowSolver, _: &Context, _: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<ValueNumberingAnalysis>()
    }

    fn visit_operation(
        this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        op: Ptr<Operation>,
        operands: &[ReadRef<SparseLattice<ValueClass>>],
        results: &[WriteRef<SparseLattice<ValueClass>>],
    ) -> Result<()> {
        if results.len() > 1 {
            this.set_all_to_entry_states(solver, ctx, results);
            return Ok(());
        }

        let has_side_effects = op
            .cast::<dyn SideEffects>(ctx)
            .is_none_or(|op| op.has_side_effects(ctx));
        let (has_innumerable_effects, mem_val) = match op.cast::<dyn MemoryEffectsOp>(ctx) {
            Some(effects) if effects.has_effects(ctx) => {
                let mut analyses = solver.analyses();
                let mut memory_ssa = MemorySSA::get_for_nearest_root(ctx, &mut analyses, op)?;
                memory_ssa.set_alias_analysis_stack(default_stack());

                let mem_val = memory_ssa.optimized_use(ctx, op);
                (mem_val.is_none(), mem_val)
            }
            Some(_) => (false, None),
            None => (true, None),
        };

        if has_side_effects || has_innumerable_effects {
            this.set_all_to_entry_states(solver, ctx, results);
            return Ok(());
        }

        let result = &results[0];
        let mut operand_values = Vec::with_capacity(operands.len());
        for operand in op.deref(ctx).operands() {
            match this.get_value_number(solver, operand) {
                None => return Ok(()),
                Some(value) => operand_values.push(ExpressionValue::new(value)),
            }
        }

        let mut expression = match op.cast::<dyn ExpressionCanonicalize>(ctx) {
            Some(canonicalize) => canonicalize.canonical_expression(ctx, operand_values),
            None => Expression::new(
                op.result(ctx).get_type(ctx),
                op.dyn_op(ctx).get_opid(),
                operand_values,
                op.deref(ctx).attributes.clone_skip_outlined(ctx),
            ),
        };
        expression.mem_value = mem_val;

        let value = ValueClass::Expression(expression);
        solver.update_state(ctx, result, |it| it.join(&value));

        Ok(())
    }

    fn set_to_entry_state(
        _this: &SparseForward<Self>,
        solver: &DataflowSolver,
        ctx: &Context,
        lattice: &WriteRef<SparseLattice<ValueClass>>,
    ) {
        solver.update_state(ctx, lattice, |it| it.join(&ValueClass::Opaque(it.anchor())));
    }
}

#[derive(Default)]
pub struct ValueNumberingAnalysis {
    next_value: Rc<Cell<u64>>,
    numbered_expressions: RefCell<HMap<NumberingKey, u64>>,
}

impl ValueNumberingAnalysis {
    fn new_value(&self) -> u64 {
        let value = self.next_value.get();
        self.next_value.update(|it| it + 1);
        value
    }

    fn update_value(
        &self,
        solver: &DataflowSolver,
        ctx: &Context,
        point: ProgramPoint,
        value: Value,
    ) {
        let class = solver.get_or_create_for::<Self, ValueClassLattice>(point, value);
        let num_lattice = solver.get_or_create_mut::<ValueNumberLattice>(value);

        let key = match class.deref().value() {
            ValueClass::Uninitialized => return,
            ValueClass::Expression(expr) => NumberingKey::Expression(expr.clone()),
            ValueClass::Opaque(value) => NumberingKey::Opaque(*value),
        };

        let mut numbered_expressions = self.numbered_expressions.borrow_mut();
        let value = *numbered_expressions
            .entry(key)
            .or_insert_with(|| self.new_value());

        solver.update_state(ctx, &num_lattice, |it| it.set(ValueNumber::new(value)));
    }
}

impl DataflowAnalysis for ValueNumberingAnalysis {
    fn verify(&self, solver: &DataflowSolver, _: &Context, _: Ptr<Operation>) -> Result<()> {
        solver.require_loaded::<SparseForward<ValueClasses>>()
    }

    fn initialize(
        &mut self,
        solver: &DataflowSolver,
        ctx: &Context,
        root: Ptr<Operation>,
    ) -> Result<()> {
        visit_all_values(
            ctx,
            &mut (self, solver),
            root,
            |ctx, (this, solver), value| {
                let point = ProgramPoint::after_value(ctx, value);
                if !is_block_live::<Self>(solver, ctx, point) {
                    return;
                }
                this.update_value(solver, ctx, point, value);
            },
        );
        Ok(())
    }

    fn visit(&self, solver: &DataflowSolver, ctx: &Context, point: ProgramPoint) -> Result<()> {
        if let Some(op) = point.prev_op(ctx) {
            for result in op.deref(ctx).results() {
                self.update_value(solver, ctx, point, result);
            }
        } else {
            let block = point.block().unwrap();
            for arg in block.deref(ctx).arguments() {
                self.update_value(solver, ctx, point, arg);
            }
        }
        Ok(())
    }
}

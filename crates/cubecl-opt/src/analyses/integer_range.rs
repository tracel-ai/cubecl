use cubecl_ir::{
    Arithmetic, Builtin, ConstantValue, ElemType, Id, Operation, Operator, Type, Value, ValueKind,
};
use hashbrown::HashMap;

use crate::{Function, GlobalState};

use super::Analysis;

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub lower_bound: Option<u64>,
    pub upper_bound: Option<u64>,
}

/// Perform analysis on the possible ranges of integer values and store the results for use in later
/// optimization passes. Reasons for integers being bounded but not constant might be: the modulo
/// operator (bounds it to `0..m`), or `UNIT_POS` (bounded by `CubeDim`). Bounds can be transferred
/// between simple arithmetic, so we can determine the possible range of a good number of values.
/// This is currently only used in index bound analysis.
#[derive(Debug, Default)]
#[allow(unused)]
pub struct Ranges {
    int_ranges: HashMap<Id, Range>,
}

impl Range {
    fn constant(val: u64) -> Self {
        Self {
            lower_bound: Some(val),
            upper_bound: Some(val),
        }
    }

    fn uint(upper: u64) -> Self {
        Self {
            lower_bound: Some(0),
            upper_bound: Some(upper),
        }
    }
}

impl Analysis for Ranges {
    fn init(opt: &mut Function, state: &GlobalState) -> Self {
        let mut this = Ranges::default();
        // Run fixed point iteration
        while this.run_loop(opt, state) {}
        this
    }
}

impl Ranges {
    fn run_loop(&mut self, func: &mut Function, state: &GlobalState) -> bool {
        for block in func.node_ids() {
            let ops = func[block].ops.clone();
            for inst in ops.borrow().values() {
                match &inst.operation {
                    Operation::Arithmetic(op) => match op {
                        Arithmetic::Add(binop) if is_uint(inst.ty()) => {
                            if let Some(out_id) = val_id(&inst.out()) {
                                let lhs_range = self.range_of(&binop.lhs);
                                let rhs_range = self.range_of(&binop.rhs);
                                let out_range = lhs_range + rhs_range;
                                if Some(&out_range) != self.int_ranges.get(&out_id) {
                                    self.int_ranges.insert(out_id, out_range);
                                    return true;
                                }
                            }
                        }
                        Arithmetic::Sub(binop) if is_uint(inst.ty()) => {
                            if let Some(out_id) = val_id(&inst.out()) {
                                let lhs_range = self.range_of(&binop.lhs);
                                let rhs_range = self.range_of(&binop.rhs);
                                let out_range = lhs_range - rhs_range;
                                if Some(&out_range) != self.int_ranges.get(&out_id) {
                                    self.int_ranges.insert(out_id, out_range);
                                    return true;
                                }
                            }
                        }
                        Arithmetic::Mul(binop) if is_uint(inst.ty()) => {
                            if let Some(out_id) = val_id(&inst.out()) {
                                let lhs_range = self.range_of(&binop.lhs);
                                let rhs_range = self.range_of(&binop.rhs);
                                let out_range = lhs_range * rhs_range;
                                if Some(&out_range) != self.int_ranges.get(&out_id) {
                                    self.int_ranges.insert(out_id, out_range);
                                    return true;
                                }
                            }
                        }
                        Arithmetic::Div(binop) if is_uint(inst.ty()) => {
                            if let Some(out_id) = val_id(&inst.out()) {
                                let lhs_range = self.range_of(&binop.lhs);
                                let rhs_range = self.range_of(&binop.rhs);
                                let out_range = lhs_range / rhs_range;
                                if Some(&out_range) != self.int_ranges.get(&out_id) {
                                    self.int_ranges.insert(out_id, out_range);
                                    return true;
                                }
                            }
                        }
                        Arithmetic::ModFloor(binop) if is_uint(inst.ty()) => {
                            if let Some(out_id) = val_id(&inst.out()) {
                                let lhs_range = self.range_of(&binop.lhs);
                                let rhs_range = self.range_of(&binop.rhs);
                                let out_range = lhs_range % rhs_range;
                                if Some(&out_range) != self.int_ranges.get(&out_id) {
                                    self.int_ranges.insert(out_id, out_range);
                                    return true;
                                }
                            }
                        }
                        _ => {}
                    },
                    Operation::Operator(Operator::ReadBuiltin(builtin)) => {
                        if let Some(out_id) = val_id(&inst.out()) {
                            let cube_dim = state.cube_dim;
                            let range = match builtin {
                                Builtin::UnitPos => Range::uint(cube_dim.num_elems() as u64 - 1),
                                Builtin::UnitPosX => Range::uint(cube_dim.x as u64 - 1),
                                Builtin::UnitPosY => Range::uint(cube_dim.y as u64 - 1),
                                Builtin::UnitPosZ => Range::uint(cube_dim.z as u64 - 1),
                                Builtin::CubeDim => Range::constant(cube_dim.num_elems() as u64),
                                Builtin::CubeDimX => Range::constant(cube_dim.x as u64),
                                Builtin::CubeDimY => Range::constant(cube_dim.y as u64),
                                Builtin::CubeDimZ => Range::constant(cube_dim.z as u64),
                                _ => Default::default(),
                            };
                            self.int_ranges.insert(out_id, range);
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }
}

impl Ranges {
    /// The possible range of values of any values, if applicable. Returns unbounded range if no range
    /// can be determined, or the type is not an integer.
    pub fn range_of(&self, val: &Value) -> Range {
        match val.kind {
            ValueKind::Value { id } if is_uint(val.ty) => {
                self.int_ranges.get(&id).copied().unwrap_or(Range {
                    lower_bound: Some(0),
                    upper_bound: None,
                })
            }
            ValueKind::Value { id } => self.int_ranges.get(&id).copied().unwrap_or_default(),
            ValueKind::Constant(ConstantValue::UInt(val)) => Range::constant(val),
            _ => Default::default(),
        }
    }
}

pub(crate) fn val_id(val: &Value) -> Option<Id> {
    match val.kind {
        ValueKind::Value { id } => Some(id),
        _ => None,
    }
}

fn is_uint(ty: Type) -> bool {
    matches!(ty.elem_type(), ElemType::UInt(_))
}

mod range_ops {
    use core::{
        fmt::Display,
        ops::{Add, Div, Mul, Rem, Sub},
    };

    use super::*;

    impl Add for Range {
        type Output = Range;

        fn add(self, rhs: Self) -> Self::Output {
            let lower_bound = self.lower_bound.zip(rhs.lower_bound);
            let upper_bound = self.upper_bound.zip(rhs.upper_bound);
            Self {
                lower_bound: lower_bound.map(|(lhs, rhs)| lhs + rhs),
                upper_bound: upper_bound.map(|(lhs, rhs)| lhs + rhs),
            }
        }
    }

    impl Sub for Range {
        type Output = Range;

        fn sub(self, rhs: Self) -> Self::Output {
            let lower_bound = self.lower_bound.zip(rhs.lower_bound);
            let upper_bound = self.upper_bound.zip(rhs.upper_bound);
            Self {
                lower_bound: lower_bound.map(|(lhs, rhs)| lhs - rhs),
                upper_bound: upper_bound.map(|(lhs, rhs)| lhs - rhs),
            }
        }
    }

    impl Mul for Range {
        type Output = Range;

        fn mul(self, rhs: Self) -> Self::Output {
            let lower_bound = self.lower_bound.zip(rhs.lower_bound);
            let upper_bound = self.upper_bound.zip(rhs.upper_bound);
            Self {
                lower_bound: lower_bound.map(|(lhs, rhs)| lhs * rhs),
                upper_bound: upper_bound.map(|(lhs, rhs)| lhs * rhs),
            }
        }
    }

    impl Div for Range {
        type Output = Range;

        fn div(self, rhs: Self) -> Self::Output {
            let lower_bound = self.lower_bound.zip(rhs.lower_bound);
            let upper_bound = self.upper_bound.zip(rhs.upper_bound);
            Self {
                lower_bound: lower_bound.map(|(lhs, rhs)| lhs.checked_div(rhs).unwrap_or(lhs)),
                upper_bound: upper_bound.map(|(lhs, rhs)| lhs.checked_div(rhs).unwrap_or(lhs)),
            }
        }
    }

    impl Rem for Range {
        type Output = Range;

        fn rem(self, rhs: Self) -> Self::Output {
            if rhs.lower_bound.is_none() || rhs.upper_bound.is_none() {
                return self;
            }
            let rhs_upper = rhs.upper_bound.unwrap();
            Range {
                lower_bound: Some(0),
                upper_bound: Some(rhs_upper - 1),
            }
        }
    }

    impl Display for Range {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match (self.lower_bound, self.upper_bound) {
                (Some(lower), Some(upper)) => write!(f, "{lower}..={upper}"),
                (None, Some(upper)) => write!(f, "..={upper}"),
                (Some(lower), None) => write!(f, "{lower}.."),
                (None, None) => write!(f, ".."),
            }
        }
    }
}

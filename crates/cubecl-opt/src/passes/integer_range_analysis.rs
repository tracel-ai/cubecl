use std::ops::{Add, Mul, Sub};

use cubecl_core::{
    ir::{Builtin, ConstantScalarValue, Operation, Operator, Variable, VariableKind},
    prelude::CubePrimitive,
};

use crate::{AtomicCounter, Optimizer, Range};

use super::OptimizerPass;

impl Range {
    fn constant(val: i64) -> Self {
        Self {
            lower_bound: Some(val),
            upper_bound: Some(val),
        }
    }

    fn uint(upper: i64) -> Self {
        Self {
            lower_bound: Some(0),
            upper_bound: Some(upper),
        }
    }
}

/// Perform analysis on the possible ranges of integer values and store the results for use in later
/// optimization passes. Reasons for integers being bounded but not constant might be: the modulo
/// operator (bounds it to `0..m`), or `UNIT_POS` (bounded by `CubeDim`). Bounds can be transferred
/// between simple arithmetic, so we can determine the possible range of a good number of variables.
/// This is currently only used in index bound analysis.
#[derive(Default, Clone, Debug)]
pub struct IntegerRangeAnalysis;

impl OptimizerPass for IntegerRangeAnalysis {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for inst in ops.borrow().values() {
                let op = match &inst.operation {
                    Operation::Operator(op) => op,
                    _ => continue,
                };
                match op {
                    Operator::Add(binop) if inst.item().elem().is_int() => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = range_of(opt, &binop.lhs);
                            let rhs_range = range_of(opt, &binop.rhs);
                            let out_range = lhs_range + rhs_range;
                            if Some(&out_range) != opt.program.int_ranges.get(&out_id) {
                                opt.program.int_ranges.insert(out_id, out_range);
                                changes.inc();
                            }
                        }
                    }
                    Operator::Sub(binop) if inst.item().elem().is_int() => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = range_of(opt, &binop.lhs);
                            let rhs_range = range_of(opt, &binop.rhs);
                            let out_range = lhs_range - rhs_range;
                            if Some(&out_range) != opt.program.int_ranges.get(&out_id) {
                                opt.program.int_ranges.insert(out_id, out_range);
                                changes.inc();
                            }
                        }
                    }
                    Operator::Mul(binop) if inst.item().elem().is_int() => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = range_of(opt, &binop.lhs);
                            let rhs_range = range_of(opt, &binop.rhs);
                            let out_range = lhs_range * rhs_range;
                            if Some(&out_range) != opt.program.int_ranges.get(&out_id) {
                                opt.program.int_ranges.insert(out_id, out_range);
                                changes.inc();
                            }
                        }
                    }
                    Operator::Div(binop) if inst.item().elem().is_int() => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = range_of(opt, &binop.lhs);
                            let rhs_range: Range = range_of(opt, &binop.rhs);
                            let out_range = lhs_range / rhs_range;
                            if Some(&out_range) != opt.program.int_ranges.get(&out_id) {
                                opt.program.int_ranges.insert(out_id, out_range);
                                changes.inc();
                            }
                        }
                    }
                    Operator::Modulo(binop) if inst.item().elem().is_int() => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = range_of(opt, &binop.lhs);
                            let rhs_range = range_of(opt, &binop.rhs);
                            let out_range = lhs_range % rhs_range;
                            if Some(&out_range) != opt.program.int_ranges.get(&out_id) {
                                opt.program.int_ranges.insert(out_id, out_range);
                                changes.inc();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// The possible range of values of any variable, if applicable. Returns unbounded range if no range
/// can be determined, or the type is not an integer.
pub(crate) fn range_of(opt: &Optimizer, var: &Variable) -> Range {
    match var.kind {
        VariableKind::Versioned { id, depth, version } if var.item.elem() == u32::as_elem() => opt
            .program
            .int_ranges
            .get(&(id, depth, version))
            .copied()
            .unwrap_or(Range {
                lower_bound: Some(0),
                upper_bound: None,
            }),
        VariableKind::Versioned { id, depth, version } => opt
            .program
            .int_ranges
            .get(&(id, depth, version))
            .copied()
            .unwrap_or_default(),
        VariableKind::LocalBinding { id, depth } if var.item.elem() == u32::as_elem() => opt
            .program
            .int_ranges
            .get(&(id, depth, 0))
            .copied()
            .unwrap_or(Range {
                lower_bound: Some(0),
                upper_bound: None,
            }),
        VariableKind::LocalBinding { id, depth } => opt
            .program
            .int_ranges
            .get(&(id, depth, 0))
            .copied()
            .unwrap_or_default(),
        VariableKind::ConstantScalar(ConstantScalarValue::Int(val, _)) => Range::constant(val),
        VariableKind::ConstantScalar(ConstantScalarValue::UInt(val, _)) => {
            Range::constant(val as i64)
        }
        VariableKind::Builtin(builtin) => match builtin {
            Builtin::UnitPos => Range::uint(opt.cube_dim.num_elems() as i64 - 1),
            Builtin::UnitPosX => Range::uint(opt.cube_dim.x as i64 - 1),
            Builtin::UnitPosY => Range::uint(opt.cube_dim.y as i64 - 1),
            Builtin::UnitPosZ => Range::uint(opt.cube_dim.z as i64 - 1),
            Builtin::CubeCount => Range::constant(opt.cube_dim.num_elems() as i64),
            Builtin::CubeCountX => Range::constant(opt.cube_dim.x as i64),
            Builtin::CubeCountY => Range::constant(opt.cube_dim.y as i64),
            Builtin::CubeCountZ => Range::constant(opt.cube_dim.z as i64),
            _ => Default::default(),
        },
        _ => Default::default(),
    }
}

pub(crate) fn var_id(var: &Variable) -> Option<(u16, u8, u16)> {
    match var.kind {
        VariableKind::Versioned { id, depth, version } => Some((id, depth, version)),
        VariableKind::LocalBinding { id, depth } => Some((id, depth, 0)),
        _ => None,
    }
}

mod range_ops {
    use std::{
        fmt::Display,
        ops::{Div, Rem},
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
            let min_neg = self.lower_bound.map(|it| it < 0).unwrap_or(true);
            let max_pos = self.upper_bound.map(|it| it > 0).unwrap_or(true);
            if rhs.lower_bound.is_none() || rhs.upper_bound.is_none() {
                return self;
            }
            let rhs_lower = rhs.lower_bound.unwrap().abs();
            let rhs_upper = rhs.upper_bound.unwrap().abs();
            let rhs_max = rhs_lower.max(rhs_upper);
            match (min_neg, max_pos) {
                (true, false) => Range {
                    lower_bound: Some(-(rhs_max - 1)),
                    upper_bound: Some(0),
                },
                (true, true) => Range {
                    lower_bound: Some(-(rhs_max - 1)),
                    upper_bound: Some(rhs_max - 1),
                },
                (false, true) => Range {
                    lower_bound: Some(0),
                    upper_bound: Some(rhs_max - 1),
                },
                _ => self,
            }
        }
    }

    impl Display for Range {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match (self.lower_bound, self.upper_bound) {
                (Some(lower), Some(upper)) => write!(f, "{lower}..={upper}"),
                (None, Some(upper)) => write!(f, "..={upper}"),
                (Some(lower), None) => write!(f, "{lower}.."),
                (None, None) => write!(f, ".."),
            }
        }
    }
}

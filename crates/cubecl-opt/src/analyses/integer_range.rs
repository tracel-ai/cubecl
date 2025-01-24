use std::{
    collections::HashMap,
    ops::{Add, Mul, Sub},
};

use cubecl_ir::{
    Arithmetic, Builtin, ConstantScalarValue, Elem, Id, Operation, Variable, VariableKind,
};

use crate::{Optimizer, VarId};

use super::Analysis;

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub lower_bound: Option<u64>,
    pub upper_bound: Option<u64>,
}

/// Perform analysis on the possible ranges of integer values and store the results for use in later
/// optimization passes. Reasons for integers being bounded but not constant might be: the modulo
/// operator (bounds it to `0..m`), or `UNIT_POS` (bounded by `CubeDim`). Bounds can be transferred
/// between simple arithmetic, so we can determine the possible range of a good number of variables.
/// This is currently only used in index bound analysis.
#[derive(Debug, Default)]
pub struct Ranges {
    int_ranges: HashMap<VarId, Range>,
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
    fn init(opt: &mut Optimizer) -> Self {
        let mut this = Ranges::default();
        // Run fixed point iteration
        while this.run_loop(opt) {}
        this
    }
}

impl Ranges {
    fn run_loop(&mut self, opt: &mut Optimizer) -> bool {
        for block in opt.node_ids() {
            let ops = opt.program[block].ops.clone();
            for inst in ops.borrow().values() {
                let op = match &inst.operation {
                    Operation::Arithmetic(op) => op,
                    _ => continue,
                };
                match op {
                    Arithmetic::Add(binop) if is_uint(inst.item().elem()) => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = self.range_of(opt, &binop.lhs);
                            let rhs_range = self.range_of(opt, &binop.rhs);
                            let out_range = lhs_range + rhs_range;
                            if Some(&out_range) != self.int_ranges.get(&out_id) {
                                self.int_ranges.insert(out_id, out_range);
                                return true;
                            }
                        }
                    }
                    Arithmetic::Sub(binop) if is_uint(inst.item().elem()) => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = self.range_of(opt, &binop.lhs);
                            let rhs_range = self.range_of(opt, &binop.rhs);
                            let out_range = lhs_range - rhs_range;
                            if Some(&out_range) != self.int_ranges.get(&out_id) {
                                self.int_ranges.insert(out_id, out_range);
                                return true;
                            }
                        }
                    }
                    Arithmetic::Mul(binop) if is_uint(inst.item().elem()) => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = self.range_of(opt, &binop.lhs);
                            let rhs_range = self.range_of(opt, &binop.rhs);
                            let out_range = lhs_range * rhs_range;
                            if Some(&out_range) != self.int_ranges.get(&out_id) {
                                self.int_ranges.insert(out_id, out_range);
                                return true;
                            }
                        }
                    }
                    Arithmetic::Div(binop) if is_uint(inst.item().elem()) => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = self.range_of(opt, &binop.lhs);
                            let rhs_range = self.range_of(opt, &binop.rhs);
                            let out_range = lhs_range / rhs_range;
                            if Some(&out_range) != self.int_ranges.get(&out_id) {
                                self.int_ranges.insert(out_id, out_range);
                                return true;
                            }
                        }
                    }
                    Arithmetic::Modulo(binop) if is_uint(inst.item().elem()) => {
                        if let Some(out_id) = var_id(&inst.out()) {
                            let lhs_range = self.range_of(opt, &binop.lhs);
                            let rhs_range = self.range_of(opt, &binop.rhs);
                            let out_range = lhs_range % rhs_range;
                            if Some(&out_range) != self.int_ranges.get(&out_id) {
                                self.int_ranges.insert(out_id, out_range);
                                return true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }
}

fn is_uint(elem: Elem) -> bool {
    matches!(elem, Elem::UInt(_))
}

impl Ranges {
    /// The possible range of values of any variable, if applicable. Returns unbounded range if no range
    /// can be determined, or the type is not an integer.
    pub fn range_of(&self, opt: &Optimizer, var: &Variable) -> Range {
        match var.kind {
            VariableKind::Versioned { id, version } if is_uint(var.item.elem()) => self
                .int_ranges
                .get(&(id, version))
                .copied()
                .unwrap_or(Range {
                    lower_bound: Some(0),
                    upper_bound: None,
                }),
            VariableKind::Versioned { id, version } => self
                .int_ranges
                .get(&(id, version))
                .copied()
                .unwrap_or_default(),
            VariableKind::LocalConst { id } if is_uint(var.item.elem()) => {
                self.int_ranges.get(&(id, 0)).copied().unwrap_or(Range {
                    lower_bound: Some(0),
                    upper_bound: None,
                })
            }
            VariableKind::LocalConst { id } => {
                self.int_ranges.get(&(id, 0)).copied().unwrap_or_default()
            }
            VariableKind::ConstantScalar(ConstantScalarValue::UInt(val, _)) => Range::constant(val),
            VariableKind::Builtin(builtin) => match builtin {
                Builtin::UnitPos => Range::uint(opt.cube_dim.num_elems() as u64 - 1),
                Builtin::UnitPosX => Range::uint(opt.cube_dim.x as u64 - 1),
                Builtin::UnitPosY => Range::uint(opt.cube_dim.y as u64 - 1),
                Builtin::UnitPosZ => Range::uint(opt.cube_dim.z as u64 - 1),
                Builtin::CubeCount => Range::constant(opt.cube_dim.num_elems() as u64),
                Builtin::CubeCountX => Range::constant(opt.cube_dim.x as u64),
                Builtin::CubeCountY => Range::constant(opt.cube_dim.y as u64),
                Builtin::CubeCountZ => Range::constant(opt.cube_dim.z as u64),
                _ => Default::default(),
            },
            _ => Default::default(),
        }
    }
}

pub(crate) fn var_id(var: &Variable) -> Option<(Id, u16)> {
    match var.kind {
        VariableKind::Versioned { id, version } => Some((id, version)),
        VariableKind::LocalConst { id } => Some((id, 0)),
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

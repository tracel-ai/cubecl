use crate::{Dialect, shared::Variable};

use super::BuiltInAttribute;

impl<D: Dialect> Variable<D> {
    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Variable::AbsolutePosBaseName => BuiltInAttribute::ThreadPositionInGrid,
            Variable::CubeCountBaseName => BuiltInAttribute::ThreadgroupsPerGrid,
            Variable::CubeDimBaseName => BuiltInAttribute::ThreadsPerThreadgroup,
            Variable::CubePosBaseName => BuiltInAttribute::ThreadgroupPositionInGrid,
            Variable::PlaneDim => BuiltInAttribute::ThreadsPerSIMDgroup,
            Variable::PlanePos => BuiltInAttribute::SIMDgroupIndexInThreadgroup,
            Variable::UnitPosBaseName => BuiltInAttribute::ThreadPositionInThreadgroup,
            Variable::UnitPos => BuiltInAttribute::ThreadIndexInThreadgroup,
            Variable::UnitPosPlane => BuiltInAttribute::ThreadIndexInSIMDgroup,
            _ => BuiltInAttribute::None,
        }
    }
}

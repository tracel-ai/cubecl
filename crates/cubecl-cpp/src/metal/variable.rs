use crate::{Dialect, shared::Variable};

use super::BuiltInAttribute;

impl<D: Dialect> Variable<D> {
    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Self::AbsolutePos => BuiltInAttribute::ThreadPositionInGrid,
            Self::CubeCount => BuiltInAttribute::ThreadgroupsPerGrid,
            Self::CubeDim => BuiltInAttribute::ThreadsPerThreadgroup,
            Self::CubePos => BuiltInAttribute::ThreadgroupPositionInGrid,
            Self::PlaneDim => BuiltInAttribute::ThreadsPerSIMDgroup,
            Self::UnitPos => BuiltInAttribute::ThreadPositionInThreadgroup,
            Self::UnitPosGlobal => BuiltInAttribute::ThreadIndexInThreadgroup,
            Self::UnitPosPlane => BuiltInAttribute::ThreadIndexInSIMDgroup,
            _ => BuiltInAttribute::None,
        }
    }
}

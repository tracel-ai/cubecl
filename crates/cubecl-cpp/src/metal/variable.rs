use crate::{Dialect, shared::Variable};

use super::BuiltInAttribute;

impl<D: Dialect> Variable<D> {
    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Self::AbsolutePosBaseName => BuiltInAttribute::ThreadPositionInGrid,
            Self::CubeCountBaseName => BuiltInAttribute::ThreadgroupsPerGrid,
            Self::CubeDimBaseName => BuiltInAttribute::ThreadsPerThreadgroup,
            Self::CubePosBaseName => BuiltInAttribute::ThreadgroupPositionInGrid,
            Self::PlaneDim => BuiltInAttribute::ThreadsPerSIMDgroup,
            Self::PlanePos => BuiltInAttribute::SIMDgroupIndexInThreadgroup,
            Self::UnitPosBaseName => BuiltInAttribute::ThreadPositionInThreadgroup,
            Self::UnitPos => BuiltInAttribute::ThreadIndexInThreadgroup,
            Self::UnitPosPlane => BuiltInAttribute::ThreadIndexInSIMDgroup,
            _ => BuiltInAttribute::None,
        }
    }
}

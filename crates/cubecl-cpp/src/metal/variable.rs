use crate::{Dialect, shared::Builtin};

use super::BuiltInAttribute;

impl<D: Dialect> Builtin<D> {
    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Builtin::AbsolutePosBaseName => BuiltInAttribute::ThreadPositionInGrid,
            Builtin::CubeCountBaseName => BuiltInAttribute::ThreadgroupsPerGrid,
            Builtin::CubeDimBaseName => BuiltInAttribute::ThreadsPerThreadgroup,
            Builtin::CubePosBaseName => BuiltInAttribute::ThreadgroupPositionInGrid,
            Builtin::PlaneDim => BuiltInAttribute::ThreadsPerSIMDgroup,
            Builtin::PlanePos => BuiltInAttribute::SIMDgroupIndexInThreadgroup,
            Builtin::UnitPosBaseName => BuiltInAttribute::ThreadPositionInThreadgroup,
            Builtin::UnitPos => BuiltInAttribute::ThreadIndexInThreadgroup,
            Builtin::UnitPosPlane => BuiltInAttribute::ThreadIndexInSIMDgroup,
            _ => BuiltInAttribute::None,
        }
    }
}

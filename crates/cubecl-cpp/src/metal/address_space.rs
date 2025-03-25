use cubecl_core::compute::{Location, Visibility};

use crate::{
    Dialect,
    shared::{Binding, Component, Variable},
};

use super::BufferAttribute;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AddressSpace {
    Constant,
    Device,
    Thread,
    ThreadGroup,
    None,
}

impl AddressSpace {
    pub fn attribute(&self) -> BufferAttribute {
        match self {
            Self::Constant | Self::Device => BufferAttribute::Buffer,
            Self::ThreadGroup => BufferAttribute::ThreadGroup,
            _ => BufferAttribute::None,
        }
    }
}

impl Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant => f.write_str("constant"),
            Self::Device => f.write_str("device"),
            Self::ThreadGroup => f.write_str("threadgroup"),
            Self::Thread => f.write_str("thread"),
            Self::None => Ok(()),
        }
    }
}

impl From<AddressSpace> for Visibility {
    fn from(val: AddressSpace) -> Self {
        match val {
            AddressSpace::Constant => Visibility::Read,
            _ => Visibility::ReadWrite,
        }
    }
}

impl<D: Dialect> From<&Binding<D>> for AddressSpace {
    fn from(value: &Binding<D>) -> Self {
        match value.vis {
            Visibility::Read => AddressSpace::Constant,
            Visibility::ReadWrite => match value.location {
                Location::Storage => AddressSpace::Device,
                Location::Cube => AddressSpace::ThreadGroup,
            },
        }
    }
}

impl<D: Dialect> From<&Variable<D>> for AddressSpace {
    fn from(value: &Variable<D>) -> Self {
        match value {
            Variable::AbsolutePos
            | Variable::AbsolutePosX
            | Variable::AbsolutePosY
            | Variable::AbsolutePosZ
            | Variable::UnitPos
            | Variable::UnitPosX
            | Variable::UnitPosY
            | Variable::UnitPosZ
            | Variable::CubePos
            | Variable::CubePosX
            | Variable::CubePosY
            | Variable::CubePosZ
            | Variable::CubeDim
            | Variable::CubeDimX
            | Variable::CubeDimY
            | Variable::CubeDimZ
            | Variable::CubeCount
            | Variable::CubeCountX
            | Variable::CubeCountY
            | Variable::CubeCountZ
            | Variable::PlaneDim
            | Variable::UnitPosPlane => AddressSpace::None,
            Variable::GlobalInputArray(..) => AddressSpace::Constant,
            Variable::GlobalOutputArray(..) => AddressSpace::Device,
                Variable::GlobalScalar(..) => {
                if value.is_const() {
                    AddressSpace::Constant
                } else {
                    AddressSpace::Device
                }
            },
            Variable::SharedMemory( .. ) => AddressSpace::ThreadGroup,
            _ => AddressSpace::Thread,
        }
    }
}


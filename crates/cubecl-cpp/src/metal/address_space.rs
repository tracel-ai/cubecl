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
    ConstDevice,
    Device,
    Thread,
    ThreadGroup,
    None,
}

impl AddressSpace {
    pub fn attribute(&self) -> BufferAttribute {
        match self {
            AddressSpace::Constant | AddressSpace::ConstDevice | AddressSpace::Device => {
                BufferAttribute::Buffer
            }
            AddressSpace::ThreadGroup => BufferAttribute::ThreadGroup,
            _ => BufferAttribute::None,
        }
    }
}

impl Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressSpace::Constant => f.write_str("constant"),
            AddressSpace::ConstDevice => f.write_str("const device"),
            AddressSpace::Device => f.write_str("device"),
            AddressSpace::ThreadGroup => f.write_str("threadgroup"),
            AddressSpace::Thread => f.write_str("thread"),
            AddressSpace::None => Ok(()),
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
            Visibility::Read => AddressSpace::ConstDevice,
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
            Variable::AbsolutePosBaseName
            | Variable::AbsolutePosX
            | Variable::AbsolutePosY
            | Variable::AbsolutePosZ
            | Variable::UnitPosBaseName
            | Variable::UnitPosX
            | Variable::UnitPosY
            | Variable::UnitPosZ
            | Variable::CubePosBaseName
            | Variable::CubePosX
            | Variable::CubePosY
            | Variable::CubePosZ
            | Variable::CubeDimBaseName
            | Variable::CubeDimX
            | Variable::CubeDimY
            | Variable::CubeDimZ
            | Variable::CubeCountBaseName
            | Variable::CubeCountX
            | Variable::CubeCountY
            | Variable::CubeCountZ
            | Variable::PlaneDim
            | Variable::UnitPosPlane => AddressSpace::None,
            Variable::GlobalInputArray(..) => AddressSpace::ConstDevice,
            Variable::GlobalOutputArray(..) => AddressSpace::Device,
            Variable::GlobalScalar(..) => {
                if value.is_const() {
                    AddressSpace::ConstDevice
                } else {
                    AddressSpace::Device
                }
            }
            Variable::SharedMemory(..) => AddressSpace::ThreadGroup,
            _ => AddressSpace::Thread,
        }
    }
}

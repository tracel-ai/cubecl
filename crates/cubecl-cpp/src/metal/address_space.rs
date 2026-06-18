use cubecl_core::prelude::Visibility;

use crate::shared;

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

impl From<Visibility> for AddressSpace {
    fn from(value: Visibility) -> Self {
        match value {
            Visibility::Read => AddressSpace::ConstDevice,
            Visibility::ReadWrite => AddressSpace::Device,
            Visibility::Uniform => AddressSpace::Constant,
        }
    }
}

impl From<shared::ty::AddressSpace> for AddressSpace {
    fn from(value: shared::ty::AddressSpace) -> Self {
        match value {
            shared::ty::AddressSpace::Global(visibility) => visibility.into(),
            shared::ty::AddressSpace::Shared => AddressSpace::ThreadGroup,
            shared::ty::AddressSpace::Local => AddressSpace::Thread,
        }
    }
}

use cubecl_core::compute::Visibility;

use crate::BufferAttribute;
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AddressSpace {
    Constant,
    Device,
    ThreadGroup,
}

impl AddressSpace {
    pub fn attribute(&self) -> BufferAttribute {
        match self {
            Self::Constant | Self::Device => BufferAttribute::Buffer,
            Self::ThreadGroup => BufferAttribute::ThreadGroup,
        }
    }
}

impl Display for AddressSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant => f.write_str("constant"),
            Self::Device => f.write_str("device"),
            Self::ThreadGroup => f.write_str("threadgroup"),
        }
    }
}

impl Into<Visibility> for AddressSpace {
    fn into(self) -> Visibility {
        match self {
            AddressSpace::Constant => Visibility::Read,
            AddressSpace::Device => Visibility::ReadWrite,
            AddressSpace::ThreadGroup => Visibility::ReadWrite,
        }
    }
}

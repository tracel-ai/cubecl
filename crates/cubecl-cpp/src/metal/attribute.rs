use derive_more::Display;
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::Context,
    derive::pliron_attr,
    r#type::TypeHandle,
};
use std::fmt::Display;

use crate::shared::ty::Uvec3Type;

pub enum BufferAttribute {
    Buffer,
    ThreadGroup,
    None,
}

impl BufferAttribute {
    pub fn indexed_fmt(&self, index: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, " [[{self}({index})]]")
    }
}

impl Display for BufferAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buffer => f.write_str("buffer"),
            Self::ThreadGroup => f.write_str("threadgroup"),
            Self::None => Ok(()),
        }
    }
}

#[pliron_attr(name = "msl.builtin", format, verifier = "succ")]
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Display)]
pub enum BuiltInAttr {
    #[display("simdgroup_index_in_threadgroup")]
    SIMDgroupIndexInThreadgroup,
    #[display("thread_index_in_simdgroup")]
    ThreadIndexInSIMDgroup,
    #[display("thread_index_in_threadgroup")]
    ThreadIndexInThreadgroup,
    #[display("thread_position_in_grid")]
    ThreadPositionInGrid,
    #[display("thread_position_in_threadgroup")]
    ThreadPositionInThreadgroup,
    #[display("threadgroup_position_in_grid")]
    ThreadgroupPositionInGrid,
    #[display("threadgroups_per_grid")]
    ThreadgroupsPerGrid,
    #[display("threads_per_simdgroup")]
    ThreadsPerSIMDgroup,
}

impl BuiltInAttr {
    pub fn ty(&self, ctx: &Context) -> TypeHandle {
        match self {
            BuiltInAttr::SIMDgroupIndexInThreadgroup
            | BuiltInAttr::ThreadIndexInSIMDgroup
            | BuiltInAttr::ThreadIndexInThreadgroup
            | BuiltInAttr::ThreadsPerSIMDgroup => {
                IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle()
            }

            BuiltInAttr::ThreadPositionInGrid
            | BuiltInAttr::ThreadPositionInThreadgroup
            | BuiltInAttr::ThreadgroupPositionInGrid
            | BuiltInAttr::ThreadgroupsPerGrid => Uvec3Type::get(ctx).to_handle(),
        }
    }
}

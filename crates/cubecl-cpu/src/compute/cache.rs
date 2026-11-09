use cubecl_common::bytes::Bytes;
use cubecl_core::server::CubeDim;
use cubecl_llvm::cpu::shared_memory::{SharedMemories, SharedMemoryBlock};
use cubecl_server::kernel::BufferIOAttr;
use cubecl_server::serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(crate = "cubecl_server::serde")]
pub struct SharedMemoryBlockLayout {
    pub size: usize,
    pub align: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(crate = "cubecl_server::serde")]
pub struct SharedMemoriesLayout {
    pub base: usize,
    pub blocks: Vec<SharedMemoryBlockLayout>,
}

impl From<&SharedMemories> for SharedMemoriesLayout {
    fn from(sm: &SharedMemories) -> Self {
        Self {
            base: sm.base,
            blocks: sm
                .blocks
                .iter()
                .map(|b| SharedMemoryBlockLayout {
                    size: b.size,
                    align: b.align,
                })
                .collect(),
        }
    }
}

impl From<SharedMemoriesLayout> for SharedMemories {
    fn from(layout: SharedMemoriesLayout) -> Self {
        Self {
            base: layout.base,
            blocks: layout
                .blocks
                .into_iter()
                .map(|b| SharedMemoryBlock {
                    size: b.size,
                    align: b.align,
                })
                .collect(),
        }
    }
}

/// Reload envelope for precompiled CPU kernels.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(crate = "cubecl_server::serde")]
pub struct CpuCacheEntry {
    pub object: Bytes,
    pub entrypoint_name: String,
    pub cube_dim: CubeDim,
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
    pub shared_memories: SharedMemoriesLayout,
    pub needs_parallelism: bool,
}

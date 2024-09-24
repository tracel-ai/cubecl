use super::Instruction;
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
#[derive(Debug, Clone)]
pub struct Body {
    pub instructions: Vec<Instruction>,
    pub shared_memories: Vec<super::SharedMemory>,
    pub const_arrays: Vec<super::ConstArray>,
    pub local_arrays: Vec<super::LocalArray>,
    pub stride: bool,
    pub shape: bool,
    pub rank: bool,
    pub wrap_size_checked: bool,
    pub settings: VariableSettings,
}

/// The settings to generate the right variables.
#[derive(Debug, Clone, Default)]
pub struct VariableSettings {
    pub idx_global: bool,
    pub thread_idx_global: bool,
    pub absolute_idx: (bool, bool, bool),
    pub block_idx_global: bool,
    pub block_dim_global: bool,
    pub grid_dim_global: bool,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.settings.idx_global
            || self.settings.absolute_idx.0
            || self.settings.absolute_idx.1
            || self.settings.absolute_idx.2
        {
            f.write_str(
                "
    int3 absoluteIdx = make_int3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    );
",
            )?;
        }

        if self.settings.idx_global {
            f.write_str(
                "
    uint idxGlobal = (absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y) + (absoluteIdx.y * gridDim.x * blockDim.x) + absoluteIdx.x;
",
            )?;
        }

        if self.settings.thread_idx_global {
            f.write_str(
                "
    int threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
            ",
            )?;
        }
        if self.settings.block_idx_global {
            f.write_str(
                "
    int blockIdxGlobal = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
            ",
            )?;
        }

        if self.settings.block_dim_global {
            f.write_str(
                "
    int blockDimGlobal = blockDim.x * blockDim.y * blockDim.z;
            ",
            )?;
        }

        if self.settings.grid_dim_global {
            f.write_str(
                "
    int gridDimGlobal = gridDim.x * gridDim.y * gridDim.z;
            ",
            )?;
        }

        if self.wrap_size_checked {
            f.write_str(
                "
 int warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
",
            )?;
        }

        if self.rank || self.stride || self.shape {
            f.write_str("uint rank = info[0];\n")?;
        }

        if self.stride || self.shape {
            f.write_str("uint rank_2 = rank * 2;\n")?;
        }

        for shared in self.shared_memories.iter() {
            f.write_fmt(format_args!(
                "__shared__ {} shared_memory_{}[{}];\n",
                shared.item, shared.index, shared.size
            ))?;
        }

        for const_array in self.const_arrays.iter() {
            f.write_fmt(format_args!(
                "const {} arrays_{}[{}] = {{",
                const_array.item, const_array.index, const_array.size
            ))?;
            for value in const_array.values.iter() {
                f.write_fmt(format_args!("{value},"))?;
            }
            f.write_str("};\n")?;
        }

        // Local arrays
        for array in self.local_arrays.iter() {
            f.write_fmt(format_args!(
                "{} l_arr_{}_{}[{}];\n\n",
                array.item, array.index, array.depth, array.size
            ))?;
        }

        for ops in self.instructions.iter() {
            f.write_fmt(format_args!("{ops}"))?;
        }

        Ok(())
    }
}

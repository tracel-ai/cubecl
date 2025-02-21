use super::{barrier::BarrierOps, pipeline::PipelineOps, Dialect, Instruction, Variable};
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
#[derive(Debug, Clone)]
pub struct Body<D: Dialect> {
    pub instructions: Vec<Instruction<D>>,
    pub shared_memories: Vec<super::SharedMemory<D>>,
    pub pipelines: Vec<PipelineOps<D>>,
    pub barriers: Vec<BarrierOps<D>>,
    pub const_arrays: Vec<super::ConstArray<D>>,
    pub local_arrays: Vec<super::LocalArray<D>>,
    pub warp_size_checked: bool,
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

impl<D: Dialect> Display for Body<D> {
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

        if self.warp_size_checked {
            f.write_str(
                "
 int warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
",
            )?;
        }

        for shared in self.shared_memories.iter() {
            writeln!(
                f,
                "__shared__ {} shared_memory_{}[{}];",
                shared.item, shared.index, shared.size
            )?;
        }

        for pipeline in self.pipelines.iter() {
            writeln!(f, "{pipeline}")?;
        }
        for barrier in self.barriers.iter() {
            writeln!(f, "{barrier}")?;
        }

        for const_array in self.const_arrays.iter() {
            f.write_fmt(format_args!(
                "const {} arrays_{}[{}] = {{",
                const_array.item, const_array.index, const_array.size
            ))?;
            let elem = const_array.item.elem;
            for value in const_array.values.iter().copied() {
                let value = match value {
                    Variable::ConstantScalar(value, _) => Variable::ConstantScalar(value, elem),
                    _ => unreachable!("Value is always constant"),
                };
                f.write_fmt(format_args!("{value},"))?;
            }
            f.write_str("};\n")?;
        }

        // Local arrays
        for array in self.local_arrays.iter() {
            write!(
                f,
                "{} l_arr_{}[{}];\n\n",
                array.item, array.index, array.size
            )?;
        }

        D::local_variables(f)?;

        for ops in self.instructions.iter() {
            write!(f, "{ops}")?;
        }

        Ok(())
    }
}

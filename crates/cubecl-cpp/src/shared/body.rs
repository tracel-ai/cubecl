use super::{Dialect, Instruction, Variable, barrier::BarrierOps, pipeline::PipelineOps};
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
}

impl<D: Dialect> Display for Body<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_bindings_body(f, self)?;

        for shared in &self.shared_memories {
            D::compile_shared_memory_declaration(f, shared)?;
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

        D::compile_wmma_local_variables(f)?;

        for ops in self.instructions.iter() {
            write!(f, "{ops}")?;
        }

        Ok(())
    }
}

use super::{Body, Extension, Item, Variable};
use cubecl_core::{ir::CubeDim, CompilerRepresentation};
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Location {
    Storage,
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Visibility {
    Read,
    ReadWrite,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding {
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory {
    location: Location,
    pub index: u16,
    item: Item,
    size: u32,
}

impl SharedMemory {
    pub fn new(index: u16, item: Item, size: u32) -> Self {
        Self {
            location: Location::Workgroup,
            index,
            item,
            size,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstantArray {
    pub index: u16,
    pub item: Item,
    pub size: u32,
    pub values: Vec<Variable>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray {
    pub index: u16,
    item: Item,
    name: u8,
    size: u32,
}

impl LocalArray {
    pub fn new(index: u16, item: Item, name: u8, size: u32) -> Self {
        Self {
            index,
            item,
            name,
            size,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeShader {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    pub shared_memories: Vec<SharedMemory>,
    pub constant_arrays: Vec<ConstantArray>,
    pub local_arrays: Vec<LocalArray>,
    pub workgroup_size: CubeDim,
    pub global_invocation_id: bool,
    pub local_invocation_index: bool,
    pub local_invocation_id: bool,
    pub num_workgroups: bool,
    pub workgroup_id: bool,
    pub subgroup_size: bool,
    pub num_workgroups_no_axis: bool,
    pub workgroup_id_no_axis: bool,
    pub workgroup_size_no_axis: bool,
    pub body: Body,
    pub extensions: Vec<Extension>,
}

impl Display for ComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::format_bindings(f, "input", &self.inputs, 0)?;
        Self::format_bindings(f, "output", &self.outputs, self.inputs.len())?;

        for (i, (name, binding)) in self.named.iter().enumerate() {
            Self::format_binding(
                f,
                name.as_str(),
                binding,
                self.inputs.len() + self.outputs.len() + i,
            )?;
        }

        for array in self.shared_memories.iter() {
            write!(
                f,
                "var<{}> shared_memory_{}: array<{}, {}>;\n\n",
                array.location, array.index, array.item, array.size
            )?;
        }

        for array in self.constant_arrays.iter() {
            write!(
                f,
                "const arrays_{}: array<{}, {}> = array(",
                array.index, array.item, array.size
            )?;
            for value in array.values.iter() {
                let value = value.fmt_cast_to(array.item);
                write!(f, "{value},")?;
            }
            f.write_str(");\n\n")?;
        }

        write!(
            f,
            "const WORKGROUP_SIZE_X = {}u;
const WORKGROUP_SIZE_Y = {}u;
const WORKGROUP_SIZE_Z = {}u;\n",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        )?;

        write!(
            f,
            "
@compute
@workgroup_size({}, {}, {})
fn main(
",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z
        )?;

        if self.global_invocation_id {
            f.write_str("    @builtin(global_invocation_id) global_id: vec3<u32>,\n")?;
        }

        if self.local_invocation_index {
            f.write_str("    @builtin(local_invocation_index) local_idx: u32,\n")?;
        }

        if self.local_invocation_id {
            f.write_str("    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,\n")?;
        }

        if self.num_workgroups {
            f.write_str("    @builtin(num_workgroups) num_workgroups: vec3<u32>,\n")?;
        }

        if self.workgroup_id {
            f.write_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n")?;
        }
        if self.subgroup_size {
            f.write_str("    @builtin(subgroup_size) subgroup_size: u32,\n")?;
        }

        // Open body
        write!(f, ") {{")?;

        // Local arrays
        for array in self.local_arrays.iter() {
            write!(
                f,
                "var a_{}_{}: array<{}, {}>;\n\n",
                array.name, array.index, array.item, array.size
            )?;
        }

        // Body
        if self.workgroup_id_no_axis {
            f.write_str("let workgroup_id_no_axis = (num_workgroups.y * num_workgroups.x * workgroup_id.z) + (num_workgroups.x * workgroup_id.y) + workgroup_id.x;\n")?;
        }

        if self.workgroup_size_no_axis {
            f.write_str("let workgroup_size_no_axis = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;\n")?;
        }

        if self.num_workgroups_no_axis {
            f.write_str("let num_workgroups_no_axis = num_workgroups.x * num_workgroups.y * num_workgroups.z;\n")?;
        }

        write!(f, "{}", self.body)?;

        // Close body
        write!(f, "}}")?;

        for extension in self.extensions.iter() {
            write!(f, "{extension}\n\n")?;
        }

        Ok(())
    }
}

impl ComputeShader {
    fn format_bindings(
        f: &mut core::fmt::Formatter<'_>,
        prefix: &str,
        bindings: &[Binding],
        num_entry: usize,
    ) -> core::fmt::Result {
        for (i, binding) in bindings.iter().enumerate() {
            Self::format_binding(
                f,
                format!("{prefix}_{i}_global").as_str(),
                binding,
                num_entry + i,
            )?;
        }

        Ok(())
    }

    fn format_binding(
        f: &mut core::fmt::Formatter<'_>,
        name: &str,
        binding: &Binding,
        num_entry: usize,
    ) -> core::fmt::Result {
        let ty = match binding.size {
            Some(size) => format!("array<{}, {}>", binding.item, size),
            None => format!("array<{}>", binding.item),
        };

        write!(
            f,
            "@group(0)
@binding({})
var<{}, {}> {}: {};
\n",
            num_entry, binding.location, binding.visibility, name, ty
        )?;

        Ok(())
    }
}

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::Storage => f.write_str("storage"),
            Location::Workgroup => f.write_str("workgroup"),
        }
    }
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(exclusive_memory_only)]
            Visibility::Read => f.write_str("read"),
            _ => f.write_str("read_write"),
        }
    }
}

impl CompilerRepresentation for ComputeShader {
    fn shared_memory_size(&self) -> usize {
        // not used in wgsl compiler
        0
    }
}

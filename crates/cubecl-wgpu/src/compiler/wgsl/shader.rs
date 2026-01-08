use super::{Body, Elem, Extension, Item, Variable};
use cubecl_core::{CubeDim, ir::Id, prelude::Visibility};
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Location {
    Storage,
    Workgroup,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding {
    pub id: Id,
    pub location: Location,
    pub visibility: Visibility,
    pub item: Item,
    pub size: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedArray {
    location: Location,
    pub index: Id,
    item: Item,
    size: u32,
    alignment: Option<u32>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedValue {
    location: Location,
    pub index: Id,
    item: Item,
}

impl SharedArray {
    pub fn new(index: Id, item: Item, size: u32, alignment: Option<u32>) -> Self {
        Self {
            location: Location::Workgroup,
            index,
            item,
            size,
            alignment,
        }
    }
}

impl SharedValue {
    pub fn new(index: Id, item: Item) -> Self {
        Self {
            location: Location::Workgroup,
            index,
            item,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstantArray {
    pub index: Id,
    pub item: Item,
    pub size: u32,
    pub values: Vec<Variable>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray {
    pub index: Id,
    item: Item,
    size: u32,
}

impl LocalArray {
    pub fn new(index: Id, item: Item, size: u32) -> Self {
        Self { index, item, size }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeShader {
    pub buffers: Vec<Binding>,
    pub scalars: Vec<(Elem, usize)>,
    pub shared_arrays: Vec<SharedArray>,
    pub shared_values: Vec<SharedValue>,
    pub constant_arrays: Vec<ConstantArray>,
    pub local_arrays: Vec<LocalArray>,
    pub has_metadata: bool,
    pub workgroup_size: CubeDim,
    pub address_type: Elem,
    pub global_invocation_id: bool,
    pub local_invocation_index: bool,
    pub local_invocation_id: bool,
    pub num_workgroups: bool,
    pub workgroup_id: bool,
    pub subgroup_size: bool,
    pub subgroup_invocation_id: bool,
    pub num_workgroups_no_axis: bool,
    pub workgroup_id_no_axis: bool,
    pub workgroup_size_no_axis: bool,
    pub body: Body,
    pub extensions: Vec<Extension>,
    pub kernel_name: String,
    pub subgroup_instructions_used: bool,
    pub f16_used: bool,
}

impl Display for ComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // On wasm, writeout what extensions we're using. This is standard wgsl but not yet
        // supported by wgpu.
        if self.subgroup_instructions_used {
            #[cfg(target_family = "wasm")]
            f.write_str("enable subgroups;")?;
        }

        if self.f16_used {
            f.write_str("enable f16;")?;
        }

        Self::format_bindings(f, "buffer", &self.buffers, 0)?;

        let mut offset = self.buffers.len();
        if self.has_metadata {
            Self::format_scalar_binding(f, "info", self.address_type, None, offset)?;
            offset += 1;
        }

        for (i, (elem, len)) in self.scalars.iter().enumerate() {
            Self::format_scalar_binding(
                f,
                &format!("scalars_{elem}"),
                *elem,
                Some(*len),
                offset + i,
            )?;
        }

        for array in self.shared_arrays.iter() {
            write!(
                f,
                "var<{}> shared_memory_{}: array<{}, {}>;\n\n",
                array.location, array.index, array.item, array.size
            )?;
        }

        for value in self.shared_values.iter() {
            write!(
                f,
                "var<{}> shared_memory_{}: {};\n\n",
                value.location, value.index, value.item,
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
fn {}(
",
            self.workgroup_size.x, self.workgroup_size.y, self.workgroup_size.z, self.kernel_name
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
        if self.subgroup_invocation_id {
            f.write_str("    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,\n")?;
        }

        // Open body
        f.write_str(") {\n")?;

        // Local arrays
        for array in self.local_arrays.iter() {
            writeln!(
                f,
                "var a_{}: array<{}, {}>;\n",
                array.index, array.item, array.size
            )?;
        }

        let addr_ty = self.address_type;

        // Body
        if self.workgroup_id_no_axis {
            writeln!(
                f,
                "let workgroup_id_no_axis = ({addr_ty}(num_workgroups.y) * {addr_ty}(num_workgroups.x) * {addr_ty}(workgroup_id.z)) + ({addr_ty}(num_workgroups.x) * {addr_ty}(workgroup_id.y)) + {addr_ty}(workgroup_id.x);"
            )?;
        }

        if self.workgroup_size_no_axis {
            f.write_str("let workgroup_size_no_axis = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;\n")?;
        }

        if self.num_workgroups_no_axis {
            writeln!(
                f,
                "let num_workgroups_no_axis = {addr_ty}(num_workgroups.x) * {addr_ty}(num_workgroups.y) * {addr_ty}(num_workgroups.z);"
            )?;
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

        let visibility = match binding.visibility {
            #[cfg(exclusive_memory_only)]
            Visibility::Read => "read",
            _ => "read_write",
        };

        write!(
            f,
            "@group(0)
@binding({})
var<{}, {}> {}: {};
\n",
            num_entry, binding.location, visibility, name, ty
        )?;

        Ok(())
    }

    fn format_scalar_binding(
        f: &mut core::fmt::Formatter<'_>,
        name: &str,
        elem: Elem,
        len: Option<usize>,
        num_entry: usize,
    ) -> core::fmt::Result {
        // Scalar bindings are separately allocated and always on their own page, so we can mark them as read-only.
        // Really, they SHOULD be marked as <uniform> but that requires an alignment of 16 bytes currently,
        // and that would complicate generating the shader code.
        let ty = match len {
            Some(size) => format!("array<{elem}, {size}>"),
            None => format!("array<{elem}>"),
        };
        let location = Location::Storage;
        let visibility = "read";

        write!(
            f,
            "@group(0)
@binding({num_entry})
var<{location}, {visibility}> {name}: {ty};
\n",
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

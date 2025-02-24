use crate::{AddressSpace, Binding, Body, Item, Variable};
use cubecl_core::{ir as cube, CubeDim};
use std::fmt::Display;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory {
    address_space: AddressSpace,
    pub index: cube::Id,
    item: Item,
    size: u32,
}

impl SharedMemory {
    pub fn new(index: cube::Id, item: Item, size: u32) -> Self {
        Self {
            address_space: AddressSpace::ThreadGroup,
            index,
            item,
            size,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstantArray {
    pub index: cube::Id,
    pub item: Item,
    pub size: u32,
    pub values: Vec<Variable>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray {
    pub index: cube::Id,
    item: Item,
    size: u32,
}

impl LocalArray {
    pub fn new(index: cube::Id, item: Item, size: u32) -> Self {
        Self { index, item, size }
    }
}

#[derive(Debug, Clone)]
pub struct MetalKernel {
    pub cube_dim: CubeDim,
    // kernel args
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub named: Vec<(String, Binding)>,
    // builtin args
    // thread position in grid
    pub thread_index_in_grid: bool,
    pub thread_position_in_grid: bool,
    // thread count in threadgroup
    pub total_threads_in_threadgroup: bool,
    pub threads_per_threadgroup: bool,
    // thread position in threadgroup
    pub thread_index_in_threadgroup: bool,
    pub thread_position_in_threadgroup: bool,
    // threadgroup count in grid
    pub total_threadgroups_in_grid: bool,
    pub threadgroups_per_grid: bool,
    // threadgroup position in grid
    pub threadgroup_index_in_grid: bool,
    pub threadgroup_position_in_grid: bool,
    // simd-groups
    pub thread_index_in_simdgroup: bool,
    pub threads_per_simdgroup: bool,


    pub shared_memories: Vec<SharedMemory>,
    pub constant_arrays: Vec<ConstantArray>,
    pub local_arrays: Vec<LocalArray>,
    pub body: Body,
    pub kernel_name: String,
}

impl Display for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // includes and namespaces -------------------------------------------
        write!(f, "
#include <metal_stdlib>
using namespace metal;
"
        )?;

        // Kernel signature --------------------------------------------------
        write!(f, "
[[kernel]]
void {}(", self.kernel_name)?;
        self.format_global_bindings_args(f)?;
        self.format_metal_builtin_bindings_args(f)?;
        write!(f, "
)")?;
        // Body --------------------------------------------------------------
        write!(f, "{{
")?;
        self.format_cube_builtin_bindings_decl(f)?;
        write!(f, "{}", self.body)?;
        write!(f, "
}}
")?;

        Ok(())
    }
}

impl MetalKernel {
    fn format_global_bindings_args(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut buffer_idx = 0;
        for (i, b) in self.inputs.iter().enumerate() {
            Self::format_global_binding_arg("in", b, Some(&i.to_string()), buffer_idx, f)?;
            buffer_idx += 1;
        }
        for (i, b) in self.outputs.iter().enumerate() {
            Self::format_global_binding_arg("out", b, Some(&i.to_string()), buffer_idx, f)?;
            buffer_idx += 1;
        }
        for (name, b) in self.named.iter() {
            Self::format_global_binding_arg(name, b, None, buffer_idx, f)?;
            buffer_idx += 1;
        }
        Ok(())
    }

    fn format_global_binding_arg(name: &str, binding: &Binding, suffix: Option<&str>, attr_idx: usize, f: &mut core::fmt::Formatter<'_>)  -> core::fmt::Result {
        let suffix = suffix.map_or("".into(), |s| format!("_{s}"));
        let (pointer, size) = match binding.size {
            Some(size) => ("".to_string(), format!("[{}]", size)),
            None => (" *".to_string(), "".to_string()),
        };

        let comma = if attr_idx > 0 { "," } else { "" };
        let address_space = binding.address_space;
        let ty = binding.item;
        let attribute = binding.address_space.attribute();

        write!(
            f,
            "{comma}\n{address_space} {ty}{pointer} g_{name}{suffix}",
        )?;
        // attribute
        attribute.indexed_fmt(attr_idx, f)?;
        write!(f, "{size}")
    }

    fn format_metal_builtin_bindings_args(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let builtins = vec![
            (self.thread_position_in_grid, Variable::ThreadPositionInGrid),
            (self.threads_per_threadgroup, Variable::ThreadsPerThreadgoup),
            (self.threadgroups_per_grid, Variable::ThreadgroupsPerGrid),
            (self.thread_index_in_threadgroup, Variable::ThreadIndexInThreadgroup),
            (self.thread_position_in_threadgroup, Variable::ThreadPositionInThreadgroup),
            (self.threadgroup_position_in_grid, Variable::ThreadgroupPositionInGrid),
            (self.thread_index_in_simdgroup, Variable::ThreadIndexInSIMDgroup),
            (self.threads_per_simdgroup, Variable::ThreadsPerSIMDgroup),
        ];
        builtins.iter().filter(|(cond, _)| *cond).try_for_each(|(_, var)| self.format_metal_builtin_binding_arg(var, f))?;
        Ok(())
    }

    fn format_metal_builtin_binding_arg(&self, variable: &Variable, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let ty = variable.item();
        let attribute = variable.attribute();
        let comma = if self.inputs.len() > 0 || self.outputs.len() > 0 || self.named.len() > 0 {
            "," } else {""};
        write!(
            f,
            "{comma}\n{ty} {variable} {attribute}",
        )?;
        Ok(())
    }

    fn format_cube_builtin_bindings_decl(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.thread_index_in_grid {
            let variable = Variable::ThreadIndexInGrid;
            let ty = variable.item();
            let thread_position_in_grid_x = Variable::ThreadPositionInGridX;
            let thread_position_in_grid_y = Variable::ThreadPositionInGridY;
            let thread_position_in_grid_z = Variable::ThreadPositionInGridZ;
            let threadgroups_per_grid_x = Variable::ThreadgroupsPerGridX;
            let threadgroups_per_grid_y = Variable::ThreadgroupsPerGridY;
            let threads_per_threadgroup_x = Variable::ThreadsPerThreadgoupX;
            let threads_per_threadgroup_y = Variable::ThreadsPerThreadgoupY;
            write!(
                f,
                "{ty} {variable} = ({thread_position_in_grid_z} * {threadgroups_per_grid_x} * {threads_per_threadgroup_x} * {threadgroups_per_grid_y} * {threads_per_threadgroup_y}) + ({thread_position_in_grid_y} * {threadgroups_per_grid_x} * {threads_per_threadgroup_x}) + {thread_position_in_grid_x};
",
            )?;
        }

        if self.total_threads_in_threadgroup {
            let variable = Variable::TotalThreadsInThreadgroup;
            let ty = variable.item();
            let threads_per_threadgroup_x = Variable::ThreadsPerThreadgoupX;
            let threads_per_threadgroup_y = Variable::ThreadsPerThreadgoupY;
            let threads_per_threadgroup_z = Variable::ThreadsPerThreadgoupZ;
            write!(
                f,
                "{ty} {variable} = {threads_per_threadgroup_x} * {threads_per_threadgroup_y} * {threads_per_threadgroup_z};
",
            )?;
        }

        if self.total_threadgroups_in_grid {
            let variable = Variable::TotalThreadgroupsInGrid;
            let ty = variable.item();
            let threadgroups_per_grid_x = Variable::ThreadgroupsPerGridX;
            let threadgroups_per_grid_y = Variable::ThreadgroupsPerGridY;
            let threadgroups_per_grid_z = Variable::ThreadgroupsPerGridZ;
            write!(
                f,
                "{ty} {variable} = {threadgroups_per_grid_x} * {threadgroups_per_grid_y} * {threadgroups_per_grid_z};
",
            )?;
        }

        if self.threadgroup_index_in_grid {
            let variable = Variable::ThreadgroupIndexInGrid;
            let ty = variable.item();
            let threadgroup_position_in_grid_x = Variable::ThreadgroupPositionInGridX;
            let threadgroup_position_in_grid_y = Variable::ThreadgroupPositionInGridY;
            let threadgroup_position_in_grid_z= Variable::ThreadgroupPositionInGridZ;
            let threadgroups_per_grid_x = Variable::ThreadgroupsPerGridX;
            let threadgroups_per_grid_y = Variable::ThreadgroupsPerGridY;
            write!(
                f,
                "{ty} {variable} = ({threadgroup_position_in_grid_z} * {threadgroups_per_grid_y} * {threadgroups_per_grid_x}) + ({threadgroup_position_in_grid_y} * {threadgroups_per_grid_x}) + {threadgroup_position_in_grid_x};
",
            )?;
        }

        Ok(())
    }

    pub fn into_wgsl_reflection(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        vec![]
    }
}

//! The builtins a kernel reads, and the arithmetic that derives one from another. Every target
//! computes them the same way from the values its hardware or its launch hands it.

use crate::prelude::*;
use cubecl_core::{self as cubecl, prelude::*};

const NB_BUILTIN: usize = 31;

/// The value each builtin reads as, once a target has computed it.
#[derive(Default)]
pub(crate) struct BuiltinValues([Option<Value>; NB_BUILTIN]);

impl BuiltinValues {
    pub(crate) fn set(&mut self, builtin: Builtin, value: Value) {
        self.0[builtin as usize] = Some(value);
    }

    pub(crate) fn get(&self, builtin: Builtin) -> Option<Value> {
        self.0[builtin as usize]
    }

    pub(crate) fn expect(&self, builtin: Builtin) -> Value {
        self.get(builtin)
            .unwrap_or_else(|| panic!("Builtin {builtin:?} should have been computed already"))
    }
}

pub(crate) struct Replacer<'a> {
    pub(crate) builtins: &'a BuiltinValues,
    pub(crate) replacements: Vec<(Value, Value)>,
}

#[cube]
pub(crate) fn constant(#[comptime] value: u32) -> u32 {
    value
}

#[cube]
pub(crate) fn unit_pos(
    unit_pos_x: u32,
    unit_pos_y: u32,
    unit_pos_z: u32,
    #[comptime] cube_dim_x: u32,
    #[comptime] cube_dim_y: u32,
) -> u32 {
    unit_pos_x + unit_pos_y * cube_dim_x + unit_pos_z * cube_dim_x * cube_dim_y
}

#[cube]
pub(crate) fn absolute_pos_x(cube_pos_x: u32, unit_pos_x: u32, #[comptime] cube_dim_x: u32) -> u32 {
    cube_pos_x * cube_dim_x + unit_pos_x
}

#[cube]
pub(crate) fn absolute_pos_y(cube_pos_y: u32, unit_pos_y: u32, #[comptime] cube_dim_y: u32) -> u32 {
    cube_pos_y * cube_dim_y + unit_pos_y
}

#[cube]
pub(crate) fn absolute_pos_z(cube_pos_z: u32, unit_pos_z: u32, #[comptime] cube_dim_z: u32) -> u32 {
    cube_pos_z * cube_dim_z + unit_pos_z
}

#[cube]
pub(crate) fn absolute_pos(cube_pos: usize, unit_pos: u32, #[comptime] cube_dim: u32) -> usize {
    cube_pos * cube_dim as usize + unit_pos as usize
}

#[cube]
pub(crate) fn cube_pos(
    cube_pos_x: u32,
    cube_pos_y: u32,
    cube_pos_z: u32,
    cube_count_x: u32,
    cube_count_y: u32,
) -> usize {
    cube_pos_z as usize * cube_count_x as usize * cube_count_y as usize
        + cube_pos_y as usize * cube_count_x as usize
        + cube_pos_x as usize
}

#[cube]
pub(crate) fn cube_count(cube_count_x: u32, cube_count_y: u32, cube_count_z: u32) -> usize {
    cube_count_x as usize * cube_count_y as usize * cube_count_z as usize
}

/// Sets the builtins a cube's shape decides: the cube and cluster dimensions, and the cluster
/// position.
pub(crate) fn set_dim_and_cluster_constants(
    scope: &Scope,
    builtins: &mut BuiltinValues,
    cube_dim: Dim3,
    cluster_dim: Dim3,
) {
    let mut set_const = |builtin: Builtin, value: u32| {
        builtins.set(builtin, constant::expand(scope, value).value(scope));
    };

    set_const(Builtin::CubeDimX, cube_dim.x);
    set_const(Builtin::CubeDimY, cube_dim.y);
    set_const(Builtin::CubeDimZ, cube_dim.z);
    set_const(Builtin::CubeDim, cube_dim.num_elems());

    set_const(Builtin::CubeClusterDimX, cluster_dim.x);
    set_const(Builtin::CubeClusterDimY, cluster_dim.y);
    set_const(Builtin::CubeClusterDimZ, cluster_dim.z);
    set_const(Builtin::CubeClusterDim, cluster_dim.num_elems());

    // No target lowers clusters, so every cube is at cluster position zero.
    set_const(Builtin::CubePosCluster, 0);
    set_const(Builtin::CubePosClusterX, 0);
    set_const(Builtin::CubePosClusterY, 0);
    set_const(Builtin::CubePosClusterZ, 0);
}

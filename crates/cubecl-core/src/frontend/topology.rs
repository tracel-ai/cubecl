//! In this file we use a trick where the constant has the same name as the module containing
//! the expand function, so that a user implicitly imports the expand function when importing the constant.

use cubecl_ir::{ExpandElement, Scope};

use super::ExpandElementTyped;

macro_rules! constant {
    ($ident:ident, $var:expr, $doc:expr) => {
        #[doc = $doc]
        pub const $ident: u32 = 2;

        #[allow(non_snake_case)]
        #[doc = $doc]
        pub mod $ident {
            use super::*;

            /// Expansion of the constant variable.
            pub fn expand(_scope: &mut Scope) -> ExpandElementTyped<u32> {
                ExpandElementTyped::new(ExpandElement::Plain(crate::ir::Variable::builtin($var)))
            }
        }
    };
}

constant!(
    PLANE_DIM,
    crate::ir::Builtin::PlaneDim,
    r"
The total amount of working units in a plane.
"
);

constant!(
    UNIT_POS_PLANE,
    crate::ir::Builtin::UnitPosPlane,
    r"
The relative position of the working unit inside the plane, without regards to cube dimensions.
"
);

constant!(
    UNIT_POS,
    crate::ir::Builtin::UnitPos,
    r"
The position of the working unit inside the cube, without regards to axis.
"
);

constant!(
    UNIT_POS_X,
    crate::ir::Builtin::UnitPosX,
    r"
The position of the working unit inside the cube along the X axis.
"
);

constant!(
    UNIT_POS_Y,
    crate::ir::Builtin::UnitPosY,
    r"
The position of the working unit inside the cube along the Y axis.
"
);

constant!(
    UNIT_POS_Z,
    crate::ir::Builtin::UnitPosZ,
    r"
The position of the working unit inside the cube along the Z axis.
"
);

constant!(
    CUBE_CLUSTER_DIM,
    crate::ir::Builtin::CubeClusterDim,
    r"
The total amount of cubes in a cluster.
"
);

constant!(
    CUBE_CLUSTER_DIM_X,
    crate::ir::Builtin::CubeClusterDimX,
    r"
The dimension of the cluster along the X axis.
"
);

constant!(
    CUBE_CLUSTER_DIM_Y,
    crate::ir::Builtin::CubeClusterDimY,
    r"
The dimension of the cluster along the Y axis.
"
);

constant!(
    CUBE_CLUSTER_DIM_Z,
    crate::ir::Builtin::CubeClusterDimZ,
    r"
The dimension of the cluster along the Z axis.
"
);

constant!(
    CUBE_DIM,
    crate::ir::Builtin::CubeDim,
    r"
The total amount of working units in a cube.
"
);

constant!(
    CUBE_DIM_X,
    crate::ir::Builtin::CubeDimX,
    r"
The dimension of the cube along the X axis.
"
);

constant!(
    CUBE_DIM_Y,
    crate::ir::Builtin::CubeDimY,
    r"
The dimension of the cube along the Y axis.
"
);

constant!(
    CUBE_DIM_Z,
    crate::ir::Builtin::CubeDimZ,
    r"
The dimension of the cube along the Z axis.
"
);

constant!(
    CUBE_POS,
    crate::ir::Builtin::CubePos,
    r"
The cube position, without regards to axis.
"
);

constant!(
    CUBE_POS_X,
    crate::ir::Builtin::CubePosX,
    r"
The cube position along the X axis.
"
);

constant!(
    CUBE_POS_Y,
    crate::ir::Builtin::CubePosY,
    r"
The cube position along the Y axis.
"
);

constant!(
    CUBE_POS_Z,
    crate::ir::Builtin::CubePosZ,
    r"
The cube position along the Z axis.
"
);

constant!(
    CUBE_POS_CLUSTER,
    crate::ir::Builtin::CubePosCluster,
    r"
The cube position within the cluster.
"
);

constant!(
    CUBE_POS_CLUSTER_X,
    crate::ir::Builtin::CubePosClusterX,
    r"
The cube position in the cluster along the X axis.
"
);

constant!(
    CUBE_POS_CLUSTER_Y,
    crate::ir::Builtin::CubePosClusterY,
    r"
The cube position in the cluster along the Y axis.
"
);

constant!(
    CUBE_POS_CLUSTER_Z,
    crate::ir::Builtin::CubePosClusterZ,
    r"
The cube position in the cluster along the Z axis.
"
);

constant!(
    CUBE_COUNT,
    crate::ir::Builtin::CubeCount,
    r"
The number of cubes launched.
"
);

constant!(
    CUBE_COUNT_X,
    crate::ir::Builtin::CubeCountX,
    r"
The number of cubes launched along the X axis.
"
);

constant!(
    CUBE_COUNT_Y,
    crate::ir::Builtin::CubeCountY,
    r"
The number of cubes launched along the Y axis.
"
);

constant!(
    CUBE_COUNT_Z,
    crate::ir::Builtin::CubeCountZ,
    r"
The number of cubes launched along the Z axis.
"
);

constant!(
    ABSOLUTE_POS,
    crate::ir::Builtin::AbsolutePos,
    r"
The position of the working unit in the whole cube kernel, without regards to cubes and axis.
"
);

constant!(
    ABSOLUTE_POS_X,
    crate::ir::Builtin::AbsolutePosX,
    r"
The index of the working unit in the whole cube kernel along the X axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Y,
    crate::ir::Builtin::AbsolutePosY,
    r"
The index of the working unit in the whole cube kernel along the Y axis, without regards to cubes.
"
);

constant!(
    ABSOLUTE_POS_Z,
    crate::ir::Builtin::AbsolutePosZ,
    r"
The index of the working unit in the whole cube kernel along the Z axis, without regards to cubes.
"
);

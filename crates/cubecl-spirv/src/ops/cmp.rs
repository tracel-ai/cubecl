use cubecl_ir::{dialect::cmp, prelude::*};
use pliron::irbuild::inserter::Inserter;
use pliron_spirv::{ext::gl, ops};

use crate::{
    ops::{
        base::{binop_to_spirv_dialect, ternop_to_spirv_dialect},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

binop_to_spirv_dialect!(cmp::SMinOp => gl::SMinOp);
binop_to_spirv_dialect!(cmp::UMinOp => gl::UMinOp);
binop_to_spirv_dialect!(cmp::FMinOp => gl::FMinOp);

binop_to_spirv_dialect!(cmp::SMaxOp => gl::SMaxOp);
binop_to_spirv_dialect!(cmp::UMaxOp => gl::UMaxOp);
binop_to_spirv_dialect!(cmp::FMaxOp => gl::FMaxOp);

ternop_to_spirv_dialect!(cmp::SClampOp => gl::SClampOp);
ternop_to_spirv_dialect!(cmp::UClampOp => gl::UClampOp);
ternop_to_spirv_dialect!(cmp::FClampOp => gl::FClampOp);

binop_to_spirv_dialect!(cmp::IEqualOp => ops::IEqualOp);
binop_to_spirv_dialect!(cmp::FEqualOp => ops::FOrdEqualOp);

binop_to_spirv_dialect!(cmp::INotEqualOp => ops::INotEqualOp);
binop_to_spirv_dialect!(cmp::FNotEqualOp => ops::FOrdNotEqualOp);

binop_to_spirv_dialect!(cmp::SGreaterThanOp => ops::SGreaterThanOp);
binop_to_spirv_dialect!(cmp::UGreaterThanOp => ops::UGreaterThanOp);
binop_to_spirv_dialect!(cmp::FGreaterThanOp => ops::FOrdGreaterThanOp);

binop_to_spirv_dialect!(cmp::SGreaterThanOrEqualOp => ops::SGreaterThanEqualOp);
binop_to_spirv_dialect!(cmp::UGreaterThanOrEqualOp => ops::UGreaterThanEqualOp);
binop_to_spirv_dialect!(cmp::FGreaterThanOrEqualOp => ops::FOrdGreaterThanEqualOp);

binop_to_spirv_dialect!(cmp::SLessThanOp => ops::SLessThanOp);
binop_to_spirv_dialect!(cmp::ULessThanOp => ops::ULessThanOp);
binop_to_spirv_dialect!(cmp::FLessThanOp => ops::FOrdLessThanOp);

binop_to_spirv_dialect!(cmp::SLessThanOrEqualOp => ops::SLessThanEqualOp);
binop_to_spirv_dialect!(cmp::ULessThanOrEqualOp => ops::ULessThanEqualOp);
binop_to_spirv_dialect!(cmp::FLessThanOrEqualOp => ops::FOrdLessThanEqualOp);

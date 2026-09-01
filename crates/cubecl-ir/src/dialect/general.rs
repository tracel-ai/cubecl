use core::cell::Ref;

use alloc::string::String;

use cubecl_macros_internal::{const_eval, cube_op, simplify};
use derive_more::From;
use derive_new::new;
use pliron::{
    builtin::attributes::{StringAttr, TypeAttr},
    derive::pliron_attr,
    r#type::type_cast,
};
use thiserror::Error;

use crate::{
    Builtin, CanMaterialize, ConstantValue, PropagatesUniformity, Pure,
    attributes::{BoolAttr, IndexAttr},
    dialect::{
        math::{index_attr, int_attr},
        pure_binop, pure_unop,
    },
    interfaces::{
        ScalarType, TriviallyUnrollable, TypedExt,
        aliasing::AliasingOp,
        uniformity::{UniformOpInterface, Uniformity},
    },
    prelude::*,
    types::scalar::IndexType,
};

#[derive(Error, Debug)]
pub enum SymbolUserOpVerifyErr {
    #[error("Symbol {0} not found")]
    SymbolNotFound(String),
    #[error("Function {0} should have been builtin.func type")]
    NotFunc(String),
    #[error("Function call has incorrect type: {0}")]
    FuncTypeErr(String),
}

#[cube_op(name = "cube.copy")]
#[result_ty(same_as = value)]
#[op_traits(Pure, CanMaterialize, PropagatesUniformity)]
pub struct CopyOp {
    pub value: Value,
}

simplify!(CopyOp, {
    |_| Some(self.value(ctx)),
});

#[op_interface_impl]
impl AliasingOp for CopyOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        Some(self.value(ctx))
    }
}

#[cube_op(name = "cube.poison")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct PoisonOp {}

pure_binop!("cube.bool_and", BoolAndOp);
const_eval!(BoolAndOp, {
    BoolAttr: |lhs, rhs| lhs && rhs,
    // false && x -> false
    custom: |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => BoolAttr::per_lane(ctx, self.get_result(ctx), false),
        _ => None
    },
    // x && false -> false
    custom: |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => BoolAttr::per_lane(ctx, self.get_result(ctx), false),
        _ => None
    }
});
simplify!(BoolAndOp, {
    // true && x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(self.rhs(ctx)),
        _ => None,
    },
    // x && true -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(self.lhs(ctx)),
        _ => None,
    },
    // x && x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_binop!("cube.bool_or", BoolOrOp);
const_eval!(BoolOrOp, {
    BoolAttr: |lhs, rhs| lhs || rhs,
    // true || x -> true
    custom: |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => BoolAttr::per_lane(ctx, self.get_result(ctx), true),
        _ => None
    },
    // x || true -> true
    custom: |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(true) => BoolAttr::per_lane(ctx, self.get_result(ctx), true),
        _ => None
    }
});
simplify!(BoolOrOp, {
    // false || x -> x
    |lhs, _| match lhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => Some(self.rhs(ctx)),
        _ => None,
    },
    // false || x -> x
    |_, rhs| match rhs?.as_const_val(ctx) {
        ConstantValue::Bool(false) => Some(self.lhs(ctx)),
        _ => None,
    },
    // x || x -> x
    |_, _| match self.lhs(ctx) == self.rhs(ctx) {
        true => Some(self.lhs(ctx)),
        false => None
    }
});

pure_unop!("cube.bool_not", BoolNotOp);
const_eval!(BoolNotOp, {
    BoolAttr: |inp| !inp
});

#[cube_op(name = "cube.cast")]
#[result_ty(argument)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize, PropagatesUniformity)]
pub struct CastOp {
    pub input: Value,
}
const_eval!(CastOp, {
    custom: |inp| {
        let val = inp?.as_const_val(ctx);
        let out_ty = self.get_result(ctx).get_type(ctx).deref(ctx);
        let elem = type_cast::<dyn ScalarType>(&*out_ty)?.elem_type(ctx);
        Some(val.cast_to(elem).as_attribute(ctx, elem))
    }
});
simplify!(CastOp, {
    |_| {
        if self.input(ctx).get_type(ctx) == self.result_type(ctx) {
            Some(self.input(ctx))
        } else {
            None
        }
    }
});

#[cube_op(name = "cube.reinterpret_cast")]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize, PropagatesUniformity)]
pub struct ReinterpretCastOp {
    pub input: Value,
}
const_eval!(ReinterpretCastOp, {
    custom: |inp| {
        // Too much weirdness around floats, don't bother dealing with it
        let val = match inp?.as_const_val(ctx) {
            ConstantValue::Int(val) => val as u64,
            ConstantValue::UInt(val) => val,
            _ => None?,
        };
        let out_ty = self.get_result(ctx).get_type(ctx);
        if out_ty.is_int(ctx) {
            Some(int_attr(ctx, out_ty, val as i128))
        } else if out_ty.is_index(ctx) {
            Some(index_attr(val as usize))
        } else {
            None
        }
    }
});
simplify!(ReinterpretCastOp, {
    |_| {
        if self.input(ctx).get_type(ctx) == self.result_type(ctx) {
            Some(self.input(ctx))
        } else {
            None
        }
    }
});

#[op_interface_impl]
impl AliasingOp for ReinterpretCastOp {
    fn source_ptr(&self, ctx: &Context) -> Option<Value> {
        Some(self.input(ctx))
    }
}

#[cube_op(name = "cube.select")]
#[result_ty(same_as = true_value)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(Pure, CanMaterialize, PropagatesUniformity)]
pub struct SelectOp {
    pub condition: Value,
    pub true_value: Value,
    pub false_value: Value,
}
simplify!(SelectOp, {
    |cond, _, _| match cond?.as_const_val(ctx) {
        ConstantValue::Bool(true) => Some(self.true_value(ctx)),
        ConstantValue::Bool(false) => Some(self.false_value(ctx)),
        _ => None,
    },
    // select(cond, x, x) -> x
    |_, _, _| match self.true_value(ctx) == self.false_value(ctx) {
        true => Some(self.true_value(ctx)),
        false => None
    }
});

#[pliron_attr(name = "cube.builtin", format, verifier = "succ")]
#[derive(new, From, PartialEq, Clone, Debug, Hash)]
pub struct BuiltinAttr(pub Builtin);

#[cube_op(
    name = "cube.read_builtin",
    format = "attr($builtin, $BuiltinAttr) ` : ` type($0)"
)]
#[result_ty(argument)]
#[op_traits(Pure, CanMaterialize)]
pub struct ReadBuiltinOp {
    pub builtin: BuiltinAttr,
}

#[op_interface_impl]
impl UniformOpInterface for ReadBuiltinOp {
    fn uniformity(&self, ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        match self.builtin(ctx).0 {
            Builtin::CubeDim
            | Builtin::CubeDimX
            | Builtin::CubeDimY
            | Builtin::CubeDimZ
            | Builtin::CubeClusterDim
            | Builtin::CubeClusterDimX
            | Builtin::CubeClusterDimY
            | Builtin::CubeClusterDimZ
            | Builtin::CubeCount
            | Builtin::CubeCountX
            | Builtin::CubeCountY
            | Builtin::CubeCountZ
            | Builtin::PlaneDim => Uniformity::Device,
            Builtin::CubePosCluster
            | Builtin::CubePosClusterX
            | Builtin::CubePosClusterY
            | Builtin::CubePosClusterZ
            | Builtin::CubePos
            | Builtin::CubePosX
            | Builtin::CubePosY
            | Builtin::CubePosZ => Uniformity::Cube,
            Builtin::PlanePos => Uniformity::Plane,
            Builtin::UnitPos
            | Builtin::UnitPosX
            | Builtin::UnitPosY
            | Builtin::UnitPosZ
            | Builtin::UnitPosPlane
            | Builtin::AbsolutePos
            | Builtin::AbsolutePosX
            | Builtin::AbsolutePosY
            | Builtin::AbsolutePosZ => Uniformity::None,
        }
    }
}

#[cube_op(name = "cube.read_scalar")]
#[result_ty(from_inputs = |ctx, ty: &TypeAttr, _| ty.get_type(ctx))]
#[op_traits(Pure, CanMaterialize)]
pub struct ReadScalarOp {
    pub ty: TypeAttr,
    pub id: IndexAttr,
}

#[op_interface_impl]
impl UniformOpInterface for ReadScalarOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::Device
    }
}

#[cube_op(name = "cube.free")]
#[result_ty(none)]
pub struct FreeOp {
    pub memory: Value,
}

#[cube_op(name = "cube.buffer_len")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct BufferLenOp {
    pub buffer_idx: IndexAttr,
}

#[op_interface_impl]
impl UniformOpInterface for BufferLenOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::Device
    }
}

#[cube_op(name = "cube.shape")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct ShapeOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}

#[op_interface_impl]
impl UniformOpInterface for ShapeOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::Device
    }
}

#[cube_op(name = "cube.stride")]
#[result_ty(fixed = IndexType::get(ctx).into())]
#[op_traits(Pure, CanMaterialize)]
pub struct StrideOp {
    pub dim: Value,
    pub buffer_idx: IndexAttr,
}

#[op_interface_impl]
impl UniformOpInterface for StrideOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::Device
    }
}

#[cube_op(name = "cube.comment")]
#[result_ty(none)]
pub struct CommentOp {
    pub comment: StringAttr,
}

#[pliron_op(name = "cube.printf", format, attributes = (cube_printf_format_string: StringAttr), verifier = "succ")]
pub struct PrintfOp;

impl PrintfOp {
    pub fn new(ctx: &mut Context, format_string: String, values: Vec<Value>) -> Self {
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        };
        op.set_attr_cube_printf_format_string(ctx, StringAttr::new(format_string));
        op
    }

    pub fn format_string<'a>(&self, ctx: &'a Context) -> Ref<'a, StringAttr> {
        self.get_attr_cube_printf_format_string(ctx).unwrap()
    }

    pub fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

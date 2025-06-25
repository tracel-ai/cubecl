use cubecl_core::{
    Feature,
    ir::{Elem, FloatKind, IntKind, UIntKind},
};
use cubecl_runtime::DeviceProperties;
use tracel_llvm::melior::{
    dialect::index,
    ir::{ValueLike, r#type::IntegerType},
};

use super::prelude::*;

impl IntoType for Elem {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        match self {
            Elem::Float(FloatKind::BF16) => Type::bfloat16(context),
            Elem::Float(FloatKind::F16) => Type::float16(context),
            Elem::Float(FloatKind::F32) => Type::float32(context),
            Elem::Float(FloatKind::F64) => Type::float64(context),
            Elem::Int(IntKind::I8) => IntegerType::new(context, 8).into(),
            Elem::Int(IntKind::I16) => IntegerType::new(context, 16).into(),
            Elem::Int(IntKind::I32) => IntegerType::new(context, 32).into(),
            Elem::Int(IntKind::I64) => IntegerType::new(context, 64).into(),
            Elem::UInt(UIntKind::U8) => IntegerType::new(context, 8).into(),
            Elem::UInt(UIntKind::U16) => IntegerType::new(context, 16).into(),
            Elem::UInt(UIntKind::U32) => IntegerType::new(context, 32).into(),
            Elem::UInt(UIntKind::U64) => IntegerType::new(context, 64).into(),
            Elem::Bool => IntegerType::new(context, 1).into(),
            _ => todo!("This type is not implemented yet."),
        }
    }
}

impl<'a> Visitor<'a> {
    pub fn visit_correct_index(
        &self,
        lhs: Value<'a, 'a>,
        rhs: Value<'a, 'a>,
    ) -> (Value<'a, 'a>, Value<'a, 'a>) {
        if lhs.r#type() == Type::index(self.context) && rhs.r#type() != Type::index(self.context) {
            let rhs = self.append_operation_with_result(index::casts(
                rhs,
                Type::index(self.context),
                self.location,
            ));
            (lhs, rhs)
        } else if lhs.r#type() != Type::index(self.context)
            && rhs.r#type() == Type::index(self.context)
        {
            let lhs = self.append_operation_with_result(index::casts(
                lhs,
                Type::index(self.context),
                self.location,
            ));
            (lhs, rhs)
        } else {
            (lhs, rhs)
        }
    }
}

pub fn register_supported_types(props: &mut DeviceProperties<Feature>) {
    let supported_types = [
        Elem::UInt(UIntKind::U8),
        Elem::UInt(UIntKind::U16),
        Elem::UInt(UIntKind::U32),
        Elem::UInt(UIntKind::U64),
        Elem::Int(IntKind::I8),
        Elem::Int(IntKind::I16),
        Elem::Int(IntKind::I32),
        Elem::Int(IntKind::I64),
        // Elem::AtomicInt(IntKind::I32),
        // Elem::AtomicInt(IntKind::I64),
        // Elem::AtomicUInt(UIntKind::U32),
        // Elem::AtomicUInt(UIntKind::U64),
        Elem::Float(FloatKind::BF16),
        Elem::Float(FloatKind::F16),
        Elem::Float(FloatKind::F32),
        Elem::Float(FloatKind::F64),
        // Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}

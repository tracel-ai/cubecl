use cubecl_core::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
use cubecl_runtime::{DeviceProperties, TypeUsage};
use tracel_llvm::mlir_rs::{
    dialect::index,
    ir::{ValueLike, r#type::IntegerType},
};

use super::prelude::*;

impl IntoType for ElemType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        match self {
            ElemType::Float(FloatKind::BF16) => Type::bfloat16(context),
            ElemType::Float(FloatKind::F16) => Type::float16(context),
            ElemType::Float(FloatKind::F32) => Type::float32(context),
            ElemType::Float(FloatKind::F64) => Type::float64(context),
            ElemType::Int(IntKind::I8) => IntegerType::new(context, 8).into(),
            ElemType::Int(IntKind::I16) => IntegerType::new(context, 16).into(),
            ElemType::Int(IntKind::I32) => IntegerType::new(context, 32).into(),
            ElemType::Int(IntKind::I64) => IntegerType::new(context, 64).into(),
            ElemType::UInt(UIntKind::U8) => IntegerType::new(context, 8).into(),
            ElemType::UInt(UIntKind::U16) => IntegerType::new(context, 16).into(),
            ElemType::UInt(UIntKind::U32) => IntegerType::new(context, 32).into(),
            ElemType::UInt(UIntKind::U64) => IntegerType::new(context, 64).into(),
            ElemType::Bool => IntegerType::new(context, 8).into(),
            _ => todo!("This type is not implemented yet. {}", self),
        }
    }
}

impl IntoType for StorageType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
        match self {
            StorageType::Scalar(ty) => ty.to_type(context),
            _ => todo!("This type is not implemented yet. {}", self),
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

pub fn register_supported_types(props: &mut DeviceProperties) {
    let supported_types = [
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        // Elem::AtomicInt(IntKind::I32),
        // Elem::AtomicInt(IntKind::I64),
        // Elem::AtomicUInt(UIntKind::U32),
        // Elem::AtomicUInt(UIntKind::U64),
        ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::F64),
        // Elem::Bool,
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all_scalar());
    }
}

use cubecl_core::ir::{self as cube, AtomicBinaryOperands, AtomicOp};
use tracel_llvm::mlir_rs::{
    dialect::{
        arith::{self, AtomicRMWKind},
        llvm::{self, CmpXchgOptions, LoadStoreOptions, attributes::AtomicOrdering, r#type},
        memref,
    },
    ir::{
        BlockLike, Value,
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
    },
};

use crate::compiler::visitor::{Visitor, prelude::IntoType};

impl<'a> Visitor<'a> {
    pub fn visit_atomic(&mut self, atomic: &AtomicOp, out: Option<cube::ExpandValue>) {
        match atomic {
            AtomicOp::Load(variable) => {
                let raw_ptr = self.get_raw_ptr(*variable);
                let size = variable.ty.elem_type().size();
                let integer_type = IntegerType::new(self.context, 64).into();
                let size = IntegerAttribute::new(integer_type, size as i64);
                let extra_options = LoadStoreOptions::new()
                    .atomic(AtomicOrdering::Unordered)
                    .align(Some(size));
                let ty = variable.ty.to_type(self.context);
                let value = self.append_operation_with_result(llvm::load(
                    self.context,
                    raw_ptr,
                    ty,
                    self.location,
                    extra_options,
                ));
                if let Some(out) = out {
                    self.insert_value(out, value);
                }
            }
            AtomicOp::Store(store_operands) => {
                let raw_ptr = self.get_raw_ptr(store_operands.ptr);
                let value = self.get_value(store_operands.value);
                let size = store_operands.value.ty.elem_type().size();
                let integer_type = IntegerType::new(self.context, 64).into();
                let size = IntegerAttribute::new(integer_type, size as i64);
                let extra_options = LoadStoreOptions::new()
                    .atomic(AtomicOrdering::Unordered)
                    .align(Some(size));
                self.block.append_operation(llvm::store(
                    self.context,
                    value,
                    raw_ptr,
                    self.location,
                    extra_options,
                ));
            }
            AtomicOp::Swap(op)
            | AtomicOp::Add(op)
            | AtomicOp::Sub(op)
            | AtomicOp::Max(op)
            | AtomicOp::Min(op)
            | AtomicOp::And(op)
            | AtomicOp::Or(op)
            | AtomicOp::Xor(op) => self.visit_atomic_binary_operands(atomic, op, out),
            AtomicOp::CompareAndSwap(compare_and_swap_operands) => {
                let ptr = self.get_raw_ptr(compare_and_swap_operands.ptr);
                let cmp = self.get_value(compare_and_swap_operands.cmp);
                let val = self.get_value(compare_and_swap_operands.val);
                let extra_options = CmpXchgOptions::new();
                let value = self.append_operation_with_result(llvm::cmpxchg(
                    self.context,
                    ptr,
                    cmp,
                    val,
                    AtomicOrdering::Monotonic,
                    AtomicOrdering::Monotonic,
                    self.location,
                    extra_options,
                ));
                if let Some(out) = out {
                    let value = self.append_operation_with_result(llvm::extract_value(
                        self.context,
                        value,
                        DenseI64ArrayAttribute::new(self.context, &[0]),
                        out.ty.to_type(self.context),
                        self.location,
                    ));
                    self.insert_value(out, value);
                }
            }
        };
    }

    fn get_raw_ptr(&mut self, value: cube::ExpandValue) -> Value<'a, 'a> {
        let value = self.get_value(value);
        let value = self.append_operation_with_result(memref::extract_aligned_pointer_as_index(
            value,
            self.location,
        ));
        let integer_ty = IntegerType::new(self.context, 64);
        let int = self.append_operation_with_result(arith::index_cast(
            value,
            integer_ty.into(),
            self.location,
        ));
        let ptr_ty = r#type::pointer(self.context, 0);
        self.append_operation_with_result(llvm::inttoptr(int, ptr_ty, self.location))
    }

    fn visit_atomic_binary_operands(
        &mut self,
        atomic: &AtomicOp,
        atomic_binary_operands: &AtomicBinaryOperands,
        out: Option<cube::ExpandValue>,
    ) {
        let ty = atomic_binary_operands.ptr.ty.elem_type();
        let value = if let AtomicOp::Sub(_) = atomic {
            self.get_neg_val(atomic_binary_operands.value)
        } else {
            self.get_value(atomic_binary_operands.value)
        };
        let kind = match atomic {
            AtomicOp::Swap(_) => AtomicRMWKind::Assign,
            AtomicOp::Add(_) if ty.is_float() => AtomicRMWKind::AddF,
            AtomicOp::Add(_) if ty.is_int() => AtomicRMWKind::AddI,
            AtomicOp::Sub(_) if ty.is_float() => AtomicRMWKind::AddF,
            AtomicOp::Sub(_) if ty.is_int() => AtomicRMWKind::AddI,
            AtomicOp::Max(_) if ty.is_float() => AtomicRMWKind::MaximumF,
            AtomicOp::Max(_) if ty.is_signed_int() => AtomicRMWKind::MaxS,
            AtomicOp::Max(_) if ty.is_unsigned_int() => AtomicRMWKind::MaxU,
            AtomicOp::Min(_) if ty.is_float() => AtomicRMWKind::MinimumF,
            AtomicOp::Min(_) if ty.is_signed_int() => AtomicRMWKind::MinS,
            AtomicOp::Min(_) if ty.is_unsigned_int() => AtomicRMWKind::MinU,
            AtomicOp::And(_) => AtomicRMWKind::AndI,
            AtomicOp::Or(_) => AtomicRMWKind::OrI,
            AtomicOp::Xor(_) => AtomicRMWKind::XOrI,
            _ => unreachable!(),
        };
        let memref = self.get_value(atomic_binary_operands.ptr);
        let zero = self.create_constant_index(0);

        let value = self.append_operation_with_result(memref::atomic_rmw(
            self.context,
            kind,
            value,
            memref,
            &[zero],
            self.location,
        ));
        if let Some(out) = out {
            self.insert_value(out, value);
        }
    }
}

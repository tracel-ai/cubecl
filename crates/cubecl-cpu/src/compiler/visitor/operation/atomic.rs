use cubecl_core::ir::{AtomicBinaryOperands, AtomicOp, Variable};
use tracel_llvm::mlir_rs::{
    dialect::{arith::AtomicRMWKind, memref},
    ir::{Attribute, Operation, attribute::StringAttribute},
};

use crate::compiler::visitor::{Visitor, operation, prelude::IntoType};

impl<'a> Visitor<'a> {
    pub fn visit_atomic(&mut self, atomic: &AtomicOp, out: Variable) {
        let operation = match atomic {
            AtomicOp::Load(variable) => todo!(),
            AtomicOp::Store(store_operands) => todo!(),
            AtomicOp::Swap(op)
            | AtomicOp::Add(op)
            | AtomicOp::Sub(op)
            | AtomicOp::Max(op)
            | AtomicOp::Min(op)
            | AtomicOp::And(op)
            | AtomicOp::Or(op)
            | AtomicOp::Xor(op) => self.visit_atomic_binary_operands(atomic, op),
            AtomicOp::CompareAndSwap(compare_and_swap_operands) => todo!(),
        };
        let value = self.append_operation_with_result(operation);
        self.insert_variable(out, value);
    }
    fn visit_atomic_binary_operands(
        &self,
        atomic: &AtomicOp,
        atomic_binary_operands: &AtomicBinaryOperands,
    ) -> Operation<'a> {
        let ty = atomic_binary_operands.ptr.ty.elem_type();
        let value = if let AtomicOp::Sub(_) = atomic {
            self.get_neg_val(atomic_binary_operands.value)
        } else {
            self.get_variable(atomic_binary_operands.value)
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
        let memref = self.get_variable(atomic_binary_operands.ptr);
        let zero = self.create_constant_index(0);

        memref::atomic_rmw(self.context, kind, value, memref, &[zero], self.location).into()
    }
}

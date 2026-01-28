pub(super) mod arithmetic;
pub(super) mod bitwise;
pub(super) mod comparison;
pub(super) mod metadata;
pub(super) mod operator;
pub(super) mod synchronization;

use cubecl_core::ir::{
    BarrierLevel, BarrierOps, NonSemantic, OpaqueType, Operation, StorageType, Synchronization,
};
use tracel_llvm::mlir_rs::{
    dialect::{llvm, ods::llvm as llvm_ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, TypeAttribute},
        r#type::IntegerType,
    },
};

use crate::compiler::visitor::prelude::*;

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operation(&mut self, operation: &Operation) {
        match operation {
            Operation::NonSemantic(NonSemantic::Print {
                format_string,
                args,
            }) => {
                let format_string = format_string.clone() + "\0";
                let global_name = self.append_global_str(&format_string);
                let str_pointer = self.append_operation_with_result(llvm_ods::mlir_addressof(
                    self.context,
                    llvm::r#type::pointer(self.context, 0),
                    FlatSymbolRefAttribute::new(self.context, &global_name),
                    self.location,
                ));
                let callee: Vec<_> = [str_pointer]
                    .into_iter()
                    .chain(args.iter().map(|arg| self.get_variable(*arg)))
                    .collect();
                let integer_type = IntegerType::new(self.context, 32).into();
                let argument_type = [llvm::r#type::pointer(self.context, 0)];
                let func_type =
                    TypeAttribute::new(llvm::r#type::function(integer_type, &argument_type, true));
                let call = llvm::call(
                    self.context,
                    FlatSymbolRefAttribute::new(self.context, "printf"),
                    Some(func_type),
                    &callee,
                    &[integer_type],
                    self.location,
                );
                self.block.append_operation(call);
            }
            // These operation are not needed in MLIR
            Operation::NonSemantic(_) => {}
            Operation::Barrier(barrier) => {
                let barrier_level = match barrier {
                    BarrierOps::Declare { barrier }
                    | BarrierOps::Init { barrier, .. }
                    | BarrierOps::InitManual { barrier, .. }
                    | BarrierOps::MemCopyAsync { barrier, .. }
                    | BarrierOps::MemCopyAsyncCooperative { barrier, .. }
                    | BarrierOps::MemCopyAsyncTx { barrier, .. }
                    | BarrierOps::Arrive { barrier }
                    | BarrierOps::ArriveTx { barrier, .. }
                    | BarrierOps::CommitCopyAsync { barrier }
                    | BarrierOps::ExpectTx { barrier, .. }
                    | BarrierOps::Wait { barrier, .. }
                    | BarrierOps::WaitParity { barrier, .. }
                    | BarrierOps::ArriveAndWait { barrier }
                    | BarrierOps::TmaLoad { barrier, .. }
                    | BarrierOps::TmaLoadIm2col { barrier, .. } => {
                        match barrier.ty.storage_type() {
                            StorageType::Opaque(OpaqueType::Barrier(level)) => Some(level),
                            _ => None,
                        }
                    }
                    BarrierOps::CopyAsync { .. } => None,
                };

                match barrier {
                    BarrierOps::ArriveAndWait { .. }
                    | BarrierOps::Wait { .. }
                    | BarrierOps::WaitParity { .. }
                    | BarrierOps::Arrive { .. }
                    | BarrierOps::ArriveTx { .. }
                    | BarrierOps::ExpectTx { .. }
                    | BarrierOps::CommitCopyAsync { .. } => {
                        if matches!(barrier_level, Some(BarrierLevel::Cube)) {
                            self.visit_synchronization(&Synchronization::SyncCube);
                        }
                    }
                    BarrierOps::Declare { .. }
                    | BarrierOps::Init { .. }
                    | BarrierOps::InitManual { .. }
                    | BarrierOps::MemCopyAsync { .. }
                    | BarrierOps::MemCopyAsyncCooperative { .. }
                    | BarrierOps::MemCopyAsyncTx { .. }
                    | BarrierOps::CopyAsync { .. }
                    | BarrierOps::TmaLoad { .. }
                    | BarrierOps::TmaLoadIm2col { .. } => {
                        // No-op: CPU execution is synchronous, so memory operations complete
                        // immediately without explicit async barriers.
                    }
                }
            }
            Operation::Synchronization(synchronization) => {
                self.visit_synchronization(synchronization);
            }
            Operation::Marker(_) => {}
            operation => {
                todo!(
                    "This operation ({}) is not implemented without an out",
                    operation
                )
            }
        }
    }

    pub fn visit_operation_with_out(&mut self, operation: &Operation, out: Variable) {
        match operation {
            Operation::Atomic(_atomic) => {
                todo!("Atomic operation are not yet supported");
            }
            Operation::Arithmetic(arithmetic) => {
                self.visit_arithmetic(arithmetic, out);
            }
            Operation::Barrier(_barrier) => {
                // Barrier results (tokens) have no representation on CPU.
            }
            Operation::Bitwise(bitwise) => {
                self.visit_bitwise(bitwise, out);
            }
            Operation::Comparison(comparison) => {
                self.visit_comparison(comparison, out);
            }
            Operation::Copy(copy) => {
                if copy.ty == out.ty {
                    let value = self.get_variable(*copy);
                    self.insert_variable(out, value);
                } else {
                    // Other backends implicitly cast on copy, so we do it too
                    // to ensure compatibility with emitted cubecl IR
                    self.visit_cast(*copy, out);
                }
            }
            Operation::Metadata(metadata) => {
                self.visit_metadata(metadata, out);
            }
            Operation::Operator(operator) => {
                self.visit_operator_with_out(operator, out);
            }
            Operation::CoopMma(_) | Operation::Plane(_) | Operation::Tma(_) => {
                panic!("{operation} is not supported on CPU.");
            }
            Operation::Branch(_) => {
                unreachable!("Branch operation are removed in SSA form");
            }
            Operation::Synchronization(_) | Operation::NonSemantic(_) | Operation::Marker(_) => {
                unreachable!("{operation} doesn't have an out");
            }
        }
    }
}

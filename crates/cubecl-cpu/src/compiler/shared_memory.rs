//! Lowering of the shared `memory.declare_variable` to a slice of the launch's shared memory.
//!
//! Every unit of a cube runs the kernel on a thread of its own, so shared memory cannot be a
//! stack allocation: all the units have to see the same bytes. The host reserves a single block
//! per launch instead (out of the stream's shared memory pool) and hands it to the kernel through
//! the [`ATTR_SHARED_MEMORY`] argument. This pass lays every shared declaration out inside that
//! block, so declaring shared memory costs nothing at runtime: it is a constant offset.
//!
//! Like the units, the cubes of a launch share the block, which is what a cube barrier is for:
//! without one, a unit racing ahead to the next cube may overwrite what the units still finishing
//! the current one are reading.

use std::cell::Cell;
use std::rc::Rc;

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::dialect::memory::DeclareVariableOp;
use cubecl_core::ir::interfaces::SizedType;
use cubecl_core::ir::prelude::*;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::dict_key;
use pliron_llvm::ops as llvm;

use crate::compiler::entrypoint::runtime_arg;

dict_key!(
    /// Marks the kernel argument pointing to the shared memory block of the launch.
    ATTR_SHARED_MEMORY, "shared_memory"
);

/// The shared memory block a kernel needs, i.e. what the host has to reserve to launch it. A
/// kernel that declares no shared memory needs no block at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemoryLayout {
    pub size: usize,
    pub align: usize,
}

impl Default for SharedMemoryLayout {
    fn default() -> Self {
        Self { size: 0, align: 1 }
    }
}

/// `(op, result, size, align)` for each shared declaration, gathered during the walk so the ops
/// can be rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
struct SharedDeclarations(Vec<(Ptr<Operation>, Value, usize, usize)>);

/// Replaces every shared declaration by its offset into the shared memory block, and publishes
/// the layout of that block through the cell it was built with.
#[derive(new)]
pub struct LowerSharedMemoryPass {
    layout: Rc<Cell<SharedMemoryLayout>>,
}

#[pass_name]
impl Pass for LowerSharedMemoryPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };

        let mut declarations = SharedDeclarations::default();
        visit_all_ops_of_type::<DeclareVariableOp, _>(
            ctx,
            &mut declarations,
            op,
            |ctx, state, d| {
                if d.addr_space(ctx).0 != AddressSpace::Shared {
                    return;
                }
                assert!(
                    d.initializer(ctx).is_none(),
                    "shared memory can't be initialized, it is uninitialized by definition"
                );
                let value_ty = d.value_ty(ctx).get_type(ctx);
                let size = {
                    let value_ty = value_ty.deref(ctx);
                    type_cast::<dyn SizedType>(&*value_ty)
                        .expect("shared memory must have a sized type")
                        .size(ctx)
                };
                state.0.push((
                    d.get_operation(),
                    d.get_result(ctx),
                    size,
                    d.alignment(ctx).0,
                ));
            },
        );

        // Leave the layout alone when there is nothing to lay out, so that a function without
        // shared memory doesn't overwrite what the kernel needs.
        if declarations.0.is_empty() {
            return Ok(res);
        }

        let block = runtime_arg(ctx, func, &ATTR_SHARED_MEMORY);
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let mut layout = SharedMemoryLayout::default();

        for (decl, result, size, align) in declarations.0 {
            // The host aligns the block by rounding its base up, which needs a power of two.
            assert!(
                align.is_power_of_two(),
                "shared memory alignment must be a power of two, got {align}"
            );
            let offset = layout.size.next_multiple_of(align);
            layout = SharedMemoryLayout {
                size: offset + size,
                align: layout.align.max(align),
            };

            let gep = llvm::GetElementPtrOp::new(
                ctx,
                block,
                vec![llvm::GepIndex::Constant(offset as u32)],
                i8_ty,
            );
            gep.get_operation().insert_before(ctx, decl);
            result.replace_all_uses_with(ctx, &gep.get_result(ctx));
            Operation::erase(decl, ctx);
        }

        self.layout.set(layout);
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

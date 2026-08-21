//! Lowering of the shared `memory.declare_variable` to a pointer the host provides.
//!
//! Every unit of a cube runs the kernel on a thread of its own, so shared memory cannot be a
//! stack allocation: all the units have to see the same bytes. The host reserves a block per
//! shared memory instead (out of the stream's shared memory pool) and passes it the same way as a
//! buffer, in the pointer table the entry ABI hands to the kernel. Declaring shared memory then
//! costs no more than reading a kernel argument.
//!
//! Like the units, the cubes of a launch share those blocks, which is what a cube barrier is for:
//! without one, a unit racing ahead to the next cube may overwrite what the units still finishing
//! the current one are reading.

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::dialect::memory::DeclareVariableOp;
use cubecl_core::ir::interfaces::SizedType;
use cubecl_core::ir::prelude::*;

use crate::shared::metadata::load_table;

/// A block of shared memory the host must reserve to launch the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemoryBlock {
    pub size: usize,
    pub align: usize,
}

/// The shared memory of a kernel: the blocks to reserve, and the slot of the pointer table their
/// pointers go to. They sit right after the buffers, which own the slots before `base`.
#[derive(Clone, Debug, Default)]
pub struct SharedMemories {
    pub base: usize,
    pub blocks: Vec<SharedMemoryBlock>,
}

/// Whether `op` declares any shared memory, i.e. whether its cube iterations share state that
/// must not be written by a unit racing ahead into the next cube.
pub fn declares_shared_memory(ctx: &Context, op: Ptr<Operation>) -> bool {
    let mut found = false;
    visit_all_ops_of_type::<DeclareVariableOp, _>(ctx, &mut found, op, |ctx, found, d| {
        *found |= d.addr_space(ctx).0 == AddressSpace::Shared;
    });
    found
}

/// `(op, result, block)` for each shared declaration, gathered during the walk so the ops can be
/// rewritten once the walker no longer holds them borrowed.
#[derive(Default)]
pub struct SharedDeclarations(Vec<(Ptr<Operation>, Value, SharedMemoryBlock)>);

impl SharedDeclarations {
    /// Collects every shared memory declared under `root`.
    pub fn collect(ctx: &Context, root: Ptr<Operation>) -> Self {
        let mut declarations = Self::default();
        visit_all_ops_of_type::<DeclareVariableOp, _>(ctx, &mut declarations, root, |ctx, s, d| {
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
            let align = d.alignment(ctx).0;
            // The host aligns a block by rounding its base up, which needs a power of two.
            assert!(
                align.is_power_of_two(),
                "shared memory alignment must be a power of two, got {align}"
            );
            let block = SharedMemoryBlock { size, align };
            s.0.push((d.get_operation(), d.get_result(ctx), block));
        });
        declarations
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Replaces every declaration by the pointer `table` holds for it, starting at slot `base`.
    /// The loads go before `before`, which must be in a block dominating every use.
    pub fn lower(
        self,
        ctx: &mut Context,
        table: Value,
        base: usize,
        before: Ptr<Operation>,
    ) -> Vec<SharedMemoryBlock> {
        self.0
            .into_iter()
            .enumerate()
            .map(|(offset, (decl, result, block))| {
                let ptr_ty = result.get_type(ctx);
                let ptr = load_table(ctx, table, base + offset, ptr_ty, before);

                result.replace_all_uses_with(ctx, &ptr);
                Operation::erase(decl, ctx);
                block
            })
            .collect()
    }
}

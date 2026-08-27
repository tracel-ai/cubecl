//! What a kernel declares as shared memory, as both targets see it.
//!
//! Collecting the declarations and measuring each block is target independent. What replaces a
//! declaration is not: see [`cpu::shared_memory`](crate::cpu::shared_memory).

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::dialect::memory::DeclareVariableOp;
use cubecl_core::ir::interfaces::SizedType;
use cubecl_core::ir::prelude::*;

/// A block of shared memory the host must reserve to launch the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemoryBlock {
    pub size: usize,
    pub align: usize,
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
pub struct SharedDeclarations(pub(crate) Vec<(Ptr<Operation>, Value, SharedMemoryBlock)>);

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
}

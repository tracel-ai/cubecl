//! Lowering of the shared memory block to LDS.
//!
//! [`AllocateSharedMemoryBlockPass`](cubecl_opt::passes::alloc_shared_memory) packs every shared
//! memory into one block of offsets. The block becomes an `external addrspace(3)` global of no
//! fixed length, so its size travels with the launch as `sharedMemBytes`, and each slice is an
//! offset into it cast to the generic address space the rest of the pipeline uses.

use cubecl_core::ir::ContextExt;
use cubecl_opt::passes::alloc_shared_memory::{AllocSharedOp, SliceSharedOp};
use pliron::builtin::ops::ModuleOp;
use pliron::identifier::Identifier;
use pliron::symbol_table::SymbolTableCollection;
use pliron_llvm::attributes::LinkageAttr;
use pliron_llvm::types::ArrayType;

use crate::shared::to_llvm::prelude::*;

/// AMDGPU's local data share, i.e. the memory a workgroup shares.
pub const LDS_ADDRESS_SPACE: u32 = 3;

/// Symbol of the one block every shared memory of the kernel is an offset into.
const LDS_BLOCK: &str = "cube_lds";

/// Bytes of LDS the kernel needs, recorded on the context as the block is lowered.
///
/// The conversion below consumes the [`AllocSharedOp`], so it cannot be read off the IR
/// afterwards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedMemorySize(pub usize);

impl CtxSharedMemory for Context {}

/// The shared memory size on the context.
pub trait CtxSharedMemory: ContextExt {
    /// Zero until a block is lowered, so this is what a kernel without shared memory reports.
    fn shared_memory_size(&self) -> usize {
        self.aux_ty::<SharedMemorySize>().0
    }
    fn set_shared_memory_size(&mut self, size: usize) {
        self.set_aux_ty(SharedMemorySize(size));
    }
}

/// Walk up from `op` to the [`ModuleOp`] it lives in, which is where the block goes.
fn parent_module(ctx: &Context, op: Ptr<Operation>) -> Option<ModuleOp> {
    let mut current = Some(op);
    while let Some(op) = current {
        if let Some(module) = Operation::get_op::<ModuleOp>(op, ctx) {
            return Some(module);
        }
        current = op.deref(ctx).get_parent_op(ctx);
    }
    None
}

/// The LDS block, declared on first use and widened to `alignment` on any later one.
///
/// `AllocateSharedMemoryBlockPass` packs a kernel's shared memories into one block, so in
/// practice there is one allocation per module. Taking the strictest alignment anyway costs
/// nothing and means a second one cannot land on a block aligned for the first.
///
/// `[0 x i8]` and `external` make it dynamically sized: the code object asks for no group
/// segment of its own and the host gives it one at launch.
fn lookup_or_insert_block(
    ctx: &mut Context,
    module: ModuleOp,
    alignment: u64,
) -> Result<Identifier> {
    let name: Identifier = LDS_BLOCK.try_into().expect("valid identifier");

    let mut symbol_tables = SymbolTableCollection::new();
    let existing = symbol_tables
        .get_symbol_table(ctx, Box::new(module))
        .lookup(&name);
    if let Some(existing) = existing {
        let existing = Operation::get_op::<llvm::GlobalOp>(existing.get_operation(), ctx)
            .expect("the LDS block is a global");
        let widened = existing.alignment(ctx).unwrap_or(0).max(alignment as u32);
        existing.set_alignment(ctx, widened);
        return Ok(name);
    }

    let byte_ty = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let block_ty = ArrayType::get(ctx, byte_ty, 0).into();
    let global = llvm::GlobalOp::new(ctx, name.clone(), block_ty);
    global.set_address_space(ctx, LDS_ADDRESS_SPACE);
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    global.set_alignment(ctx, alignment as u32);
    symbol_tables
        .get_symbol_table(ctx, Box::new(module))
        .insert(ctx, Box::new(global), None)?;

    Ok(name)
}

#[op_interface_impl]
impl ToLLVMDialect for AllocSharedOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let module = parent_module(ctx, old_op).expect("alloc_shared must be inside a module");
        let alignment = self.alignment(ctx).0 as u64;

        // Taken here: the rewrite below is what removes the op. One block per kernel, so
        // the last one written is the one the launch reserves.
        let size = self.size(ctx).0;
        ctx.set_shared_memory_size(size);

        let name = lookup_or_insert_block(ctx, module, alignment)?;

        let block = llvm::AddressOfOp::new(ctx, name, LDS_ADDRESS_SPACE);
        let address = insert(ctx, rewriter, &block);
        rewriter.replace_operation_with_values(ctx, old_op, vec![address]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for SliceSharedOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let block = self.block(ctx);
        let offset = self.offset(ctx).0 as u32;

        let byte_ty = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let gep =
            llvm::GetElementPtrOp::new(ctx, block, vec![llvm::GepIndex::Constant(offset)], byte_ty);
        let slice = insert(ctx, rewriter, &gep);

        // Back to the generic address space every load and store downstream expects.
        // `InferAddressSpaces` folds it away.
        let generic_ty = LlvmPointerType::get(ctx, 0).into();
        let op = llvm::AddrSpaceCastOp::new(ctx, slice, generic_ty);
        let generic = insert(ctx, rewriter, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![generic]);
        Ok(())
    }
}

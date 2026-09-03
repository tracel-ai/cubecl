//! What a kernel declares as shared memory, and the one block a GPU launch reserves for it.
//!
//! Collecting the declarations and measuring each block is target independent. What replaces a
//! declaration depends on the target: the CPU gives each one a slot in a per-cube arena (see
//! [`cpu::shared_memory`](crate::cpu::shared_memory)), while both GPUs pack them into a single
//! dynamically sized block, which is what the second half of this module lowers.

use cubecl_core::ir::AddressSpace;
use cubecl_core::ir::dialect::memory::DeclareVariableOp;
use cubecl_core::ir::interfaces::SizedType;
use cubecl_core::ir::prelude::*;
use cubecl_opt::passes::alloc_shared_memory::{AllocSharedOp, SliceSharedOp};
use pliron::builtin::ops::ModuleOp;
use pliron::identifier::Identifier;
use pliron::symbol_table::SymbolTableCollection;
use pliron_llvm::attributes::LinkageAttr;
use pliron_llvm::types::ArrayType;

use crate::shared::to_llvm::prelude::*;

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

/// The address space a cube's shared memory lives in. Address space 3 is AMDGPU's local
/// data share and NVPTX's `.shared`, so the one constant serves both.
pub const SHARED_ADDRESS_SPACE: u32 = 3;

/// Symbol of the one block every shared memory of the kernel is an offset into.
const SHARED_BLOCK: &str = "cube_shared";

/// Bytes of shared memory the kernel needs, recorded on the context as the block is lowered.
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

/// The shared block, declared on first use and widened to `alignment` on any later one.
///
/// `AllocateSharedMemoryBlockPass` packs a kernel's shared memories into one block, so in
/// practice there is one allocation per module. Taking the strictest alignment anyway costs
/// nothing and means a second one cannot land on a block aligned for the first.
///
/// `[0 x i8]` and `external` make it dynamically sized: the module asks for no shared
/// memory of its own and the host gives it a block at launch -- `sharedMemBytes` to
/// `hipModuleLaunchKernel`, and the same to `cuLaunchKernel`.
fn lookup_or_insert_block(
    ctx: &mut Context,
    module: ModuleOp,
    alignment: u64,
) -> Result<Identifier> {
    let name: Identifier = SHARED_BLOCK.try_into().expect("valid identifier");

    let mut symbol_tables = SymbolTableCollection::new();
    let existing = symbol_tables
        .get_symbol_table(ctx, Box::new(module))
        .lookup(&name);
    if let Some(existing) = existing {
        let existing = Operation::get_op::<llvm::GlobalOp>(existing.get_operation(), ctx)
            .expect("the shared block is a global");
        let widened = existing.alignment(ctx).unwrap_or(0).max(alignment as u32);
        existing.set_alignment(ctx, widened);
        return Ok(name);
    }

    let byte_ty = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let block_ty = ArrayType::get(ctx, byte_ty, 0).into();
    let global = llvm::GlobalOp::new(ctx, name.clone(), block_ty);
    global.set_address_space(ctx, SHARED_ADDRESS_SPACE);
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

        let block = llvm::AddressOfOp::new(ctx, name, SHARED_ADDRESS_SPACE);
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
        // `InferAddressSpaces` folds it away on both targets.
        let generic_ty = LlvmPointerType::get(ctx, 0).into();
        let op = llvm::AddrSpaceCastOp::new(ctx, slice, generic_ty);
        let generic = insert(ctx, rewriter, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![generic]);
        Ok(())
    }
}

//! Shared memory allocation and layout.

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

/// Shared memory block required for a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemoryBlock {
    pub size: usize,
    pub align: usize,
}

pub fn declares_shared_memory(ctx: &Context, op: Ptr<Operation>) -> bool {
    let mut found = false;
    visit_all_ops_of_type::<DeclareVariableOp, _>(ctx, &mut found, op, |ctx, found, d| {
        *found |= d.addr_space(ctx).0 == AddressSpace::Shared;
    });
    found
}

#[derive(Default)]
pub struct SharedDeclarations(pub(crate) Vec<(Ptr<Operation>, Value, SharedMemoryBlock)>);

impl SharedDeclarations {
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
            // Block alignment must be a power of two.
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

/// Shared address space on AMDGPU and NVPTX.
pub const SHARED_ADDRESS_SPACE: u32 = 3;

const SHARED_BLOCK: &str = "cube_shared";

/// Shared memory required per launch, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedMemorySize(pub usize);

impl CtxSharedMemory for Context {}

pub trait CtxSharedMemory: ContextExt {
    fn shared_memory_size(&self) -> usize {
        self.aux_ty::<SharedMemorySize>().0
    }
    fn set_shared_memory_size(&mut self, size: usize) {
        self.set_aux_ty(SharedMemorySize(size));
    }
}

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

/// Dynamic shared memory is provided by the host at launch.
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

        let generic_ty = LlvmPointerType::get(ctx, 0).into();
        let op = llvm::AddrSpaceCastOp::new(ctx, slice, generic_ty);
        let generic = insert(ctx, rewriter, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![generic]);
        Ok(())
    }
}

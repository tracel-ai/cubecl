use cubecl_core::WgpuCompilationOptions;
use cubecl_ir::{
    AddressSpace,
    dialect::memory::{self, DeclareVariableOp, IndexOp},
    ident,
    interfaces::TypedExt,
    prelude::*,
};
use cubecl_opt::passes::alloc_shared_memory::SliceSharedOp;
use pliron::{
    builtin::ops::{ConstantOp, FuncOp},
    common_traits::Named,
    debug_info::set_operation_result_name,
    identifier::Identifier,
    irbuild::listener::DummyListener,
    opts::dce::dce,
};
use pliron_spirv::{
    decorations::{DecoratableOp, DecorationInfo},
    ops::{self, AddressOfOp, GlobalVariableOp, InBoundsAccessChainOp, SpirvModuleOp, VariableOp},
    types::{PointerType, StructType},
};
use rspirv::spirv::{Capability, Decoration, MemoryAccess, StorageClass};

use crate::{
    attributes::attr_to_spirv_dialect,
    ops::{builtin::const_op_int32, to_spirv_dialect::ToSpirvDialectOp},
    types::{ty_to_spirv_dialect, ty_to_spirv_dialect_explicit_layout},
};

#[op_interface_impl]
impl ToSpirvDialectOp for DeclareVariableOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        assert_eq!(self.addr_space(ctx).0, AddressSpace::Local, "TODO");

        let op = self.get_operation();
        let func = find_parent_func(ctx, op);
        rewriter.set_insertion_point_to_block_start(func.get_entry_block(ctx));

        let init = self.initializer(ctx).map(|it| it.clone());
        let init = init.map(|attr| {
            let attr = attr_to_spirv_dialect(ctx, &attr);
            let constant = ConstantOp::new(ctx, attr);
            rewriter.append_op(ctx, &constant);
            constant.get_result(ctx)
        });
        let result_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let var = VariableOp::new(ctx, result_ty, StorageClass::Function, init);
        rewriter.append_op(ctx, &var);
        rewriter.replace_operation(ctx, op, var.get_operation());

        Ok(())
    }
}

fn find_parent_func(ctx: &Context, op: Ptr<Operation>) -> FuncOp {
    let mut op = op;
    while !op.is_op::<FuncOp>(ctx) {
        let parent = op.deref(ctx).get_parent_op(ctx);
        op = parent.expect("Should have parent");
    }
    op.as_op::<FuncOp>(ctx).unwrap()
}

#[op_interface_impl]
impl ToSpirvDialectOp for IndexOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = self.base(ctx);
        let index = self.index(ctx);
        let result_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let access_chain = InBoundsAccessChainOp::new(ctx, result_ty, base, vec![index]);
        rewriter.append_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::LoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let align = self.get_result(ctx).align(ctx) as u32;
        let ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let access_chain = ops::LoadOp::new(ctx, ty, ptr, MemoryAccess::ALIGNED, Some(align));
        rewriter.append_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::StoreOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let align = value.align(ctx) as u32;
        let access_chain = ops::StoreOp::new(ctx, ptr, value, MemoryAccess::ALIGNED, Some(align));
        rewriter.append_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::CopyOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let src = self.source(ctx);
        let dest = self.destination(ctx);
        let align = src.unwrap_ptr(ctx).align(ctx) as u32;
        let access_chain = ops::CopyMemoryOp::new(
            ctx,
            dest,
            src,
            MemoryAccess::ALIGNED,
            Some(align),
            MemoryAccess::NONE,
            None,
        );
        rewriter.append_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

pub fn lower_shared(ctx: &mut Context, module: SpirvModuleOp) -> (usize, Vec<Identifier>) {
    let opts = ctx.aux_ty::<WgpuCompilationOptions>();
    let op = module.get_operation();

    let mut shared_slices = vec![];
    visit_all_ops_of_type::<SliceSharedOp, _>(ctx, &mut shared_slices, op, |_, slices, op| {
        slices.push(op);
    });

    if shared_slices.is_empty() {
        return (0, vec![]);
    }

    let mut rewriter = IRRewriter::default();
    let res = if opts.vulkan.supports_explicit_smem {
        lower_shared_explicit(ctx, &mut rewriter, module, shared_slices)
    } else {
        lower_shared_implicit(ctx, &mut rewriter, module, shared_slices)
    };
    dce(op, ctx).unwrap();
    res
}

fn lower_shared_implicit(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<DummyListener>,
    module: SpirvModuleOp,
    shared_slices: Vec<SliceSharedOp>,
) -> (usize, Vec<Identifier>) {
    let mut shared_size = 0;
    let mut entry_args = vec![];
    for slice in shared_slices {
        set_operation_result_name(ctx, slice.get_operation(), 0, Some(ident("_spirv_shared")));
        let ty = ty_to_spirv_dialect(ctx, slice.value_ty(ctx).get_type(ctx));
        let value_ty = ty_to_spirv_dialect(ctx, slice.result_type(ctx));
        let name = slice.get_result(ctx).unique_name(ctx);
        shared_size += slice.value_ty(ctx).size(ctx);
        entry_args.push(name.clone());
        let var = GlobalVariableOp::new(ctx, value_ty, StorageClass::Workgroup, name.clone(), None);
        rewriter.set_insertion_point_to_block_start(module.get_body(ctx, 0));
        rewriter.append_op(ctx, &var);

        rewriter.set_insertion_point_before_operation(slice.get_operation());
        let addr = AddressOfOp::new(ctx, ty, name);
        rewriter.append_op(ctx, &addr);
        rewriter.replace_operation(ctx, slice.get_operation(), addr.get_operation());
    }
    (shared_size, entry_args)
}

fn lower_shared_explicit(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<DummyListener>,
    module: SpirvModuleOp,
    shared_slices: Vec<SliceSharedOp>,
) -> (usize, Vec<Identifier>) {
    module.insert_capability(ctx, Capability::WorkgroupMemoryExplicitLayoutKHR);

    let mut shared_size = 0;
    let mut entry_args = vec![];
    for slice in shared_slices {
        let value_ty = slice.value_ty(ctx).get_type(ctx);
        set_operation_result_name(ctx, slice.get_operation(), 0, Some(ident("_spirv_shared")));
        let name = slice.get_result(ctx).unique_name(ctx);
        shared_size += value_ty.size(ctx);
        entry_args.push(name.clone());

        let value_ty = ty_to_spirv_dialect_explicit_layout(ctx, value_ty);
        let value_ptr = PointerType::get(ctx, value_ty, StorageClass::Workgroup).to_handle();
        let offset = slice.offset(ctx).0 as u32;
        let block_dec = DecorationInfo::unit(Decoration::Block);
        let block =
            StructType::get(ctx, vec![value_ty], vec![offset], vec![], vec![block_dec]).into();
        let block_ptr = PointerType::get(ctx, block, StorageClass::Workgroup).to_handle();

        let var = GlobalVariableOp::new(ctx, block, StorageClass::Workgroup, name.clone(), None);
        var.set_decoration_aliased(ctx);
        rewriter.set_insertion_point_to_block_start(module.get_body(ctx, 0));
        rewriter.append_op(ctx, &var);

        rewriter.set_insertion_point_before_operation(slice.get_operation());
        let addr = AddressOfOp::new(ctx, block_ptr, name);
        let zero = const_op_int32(ctx, 0);
        let chain = InBoundsAccessChainOp::new(
            ctx,
            value_ptr,
            addr.get_result(ctx),
            vec![zero.get_result(ctx)],
        );
        rewriter.append_op(ctx, &addr);
        rewriter.append_op(ctx, &zero);
        rewriter.append_op(ctx, &chain);
        rewriter.replace_operation(ctx, slice.get_operation(), chain.get_operation());
    }
    (shared_size, entry_args)
}

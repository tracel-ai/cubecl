use crate::{prelude::*, shared::to_llvm::ty::padded_field_index};
use cubecl_core::ir::{
    dialect::memory::{DeclareVariableOp, IndexOp, LoadOp, StoreOp, StructFieldOp},
    settings::NarrowBindings,
    types::barrier::BarrierType,
};

#[op_interface_impl]
impl ToLLVMDialect for DeclareVariableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value_ty = self.value_ty(ctx).get_type(ctx);

        if value_ty.deref(ctx).is::<BarrierType>() {
            let useless = insert_i32_const(ctx, rewriter, 0);
            rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![useless]);
            return Ok(());
        }

        let elem_ty = cube_type_to_llvm(ctx, value_ty);

        let entry_block = enclosing_entry_block(ctx, self.get_operation());
        let insertion_point = rewriter.get_insertion_point();
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockStart(entry_block));

        let size = insert_i32_const(ctx, rewriter, 1);

        let alloca = llvm::AllocaOp::new(ctx, elem_ty, size, 0);
        let size_op = size.defining_op().expect("constant defines its result");
        rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(size_op));
        rewriter.insert_op(ctx, &alloca);

        rewriter.set_insertion_point(insertion_point);

        let initializer = self.initializer(ctx).map(|initializer| initializer.clone());
        if let Some(initializer) = initializer {
            let constant = constant_op(ctx, initializer);
            rewriter.insert_operation(ctx, constant);

            let store = llvm::StoreOp::new(ctx, constant.result(ctx), alloca.get_result(ctx));
            store.set_alignment(ctx, self.alignment(ctx).0 as u32);
            rewriter.insert_op(ctx, &store);
        }

        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![alloca.get_result(ctx)],
        );
        Ok(())
    }
}

fn enclosing_entry_block(ctx: &Context, op: Ptr<Operation>) -> Ptr<BasicBlock> {
    let mut op = op;
    loop {
        op = op
            .deref(ctx)
            .get_parent_op(ctx)
            .expect("declaration must be inside a function");
        if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
            return func.get_entry_block(ctx);
        }
    }
}

/// Unsigned cube indices require zero extension before signed GEP indexing.
fn widen_gep_index(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    index: Value,
) -> Value {
    let width = index
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|int| int.width());
    if width.is_none_or(|width| width >= GEP_INDEX_WIDTH) {
        return index;
    }

    let wide_ty = IntegerType::get(ctx, GEP_INDEX_WIDTH, Signedness::Signless).into();
    let zext = llvm::ZExtOp::new_with_nneg(ctx, index, wide_ty, false);
    rewriter.insert_op(ctx, &zext);
    zext.get_result(ctx)
}

/// The bytes an LLVM value of `ty` takes in memory, where that is a whole number of bytes: none
/// for `i1`, whose vectors pack eight values to a byte.
fn whole_bytes(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    let ty = ty.deref(ctx);
    if let Some(int) = ty.downcast_ref::<IntegerType>() {
        return int.width().is_multiple_of(8).then(|| int.width() / 8);
    }
    if ty.is::<FP16Type>() || ty.is::<BF16Type>() {
        return Some(2);
    }
    if ty.is::<FP32Type>() {
        return Some(4);
    }
    if ty.is::<FP64Type>() {
        return Some(8);
    }
    let vector = ty.downcast_ref::<LlvmVectorType>()?;
    Some(whole_bytes(ctx, vector.elem_type())? * vector.num_elements())
}

/// The shift that turns an index into `elem_ty`, an LLVM type, into a byte offset, where its size
/// is a power of two bytes, so that its LLVM alloc size is its size.
fn byte_shift(ctx: &Context, elem_ty: TypeHandle) -> Option<u32> {
    let bytes = whole_bytes(ctx, elem_ty)?;
    bytes.is_power_of_two().then(|| bytes.trailing_zeros())
}

/// Whether `address_space` is a binding the launch marked as [narrow](NarrowBindings).
fn narrow_binding(ctx: &Context, address_space: AddressSpace) -> bool {
    let AddressSpace::Global(position) = address_space else {
        return false;
    };
    ctx.try_aux_ty::<NarrowBindings>()
        .is_some_and(|bindings| bindings.contains_position(position))
}

/// `index`, a 32-bit element index, as the 64-bit byte offset `zext(index << shift)`.
///
/// The shift is `nuw`: the binding is narrow, so every byte offset into it fits 32 bits. The
/// extension carries no `nneg`, since an offset can reach 2^32 - 1.
fn byte_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    index: Value,
    shift: u32,
) -> Value {
    let shift = insert_i32_const(ctx, rewriter, shift as i32);
    let no_unsigned_wrap = IntegerOverflowFlagsAttr {
        nsw: false,
        nuw: true,
    };
    let bytes = llvm::ShlOp::new_with_overflow_flag(ctx, index, shift, no_unsigned_wrap);
    rewriter.insert_op(ctx, &bytes);
    let wide_ty = IntegerType::get(ctx, GEP_INDEX_WIDTH, Signedness::Signless).into();
    let zext = llvm::ZExtOp::new_with_nneg(ctx, bytes.get_result(ctx), wide_ty, false);
    rewriter.insert_op(ctx, &zext);
    zext.get_result(ctx)
}

#[op_interface_impl]
impl ToLLVMDialect for IndexOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = self.base(ctx);
        let index = self.index(ctx);

        let (elem_ty, address_space) = {
            let result_ty = self.get_result(ctx).get_type(ctx);
            let result_ty = result_ty.deref(ctx);
            let ptr = result_ty
                .downcast_ref::<CubePointerType>()
                .expect("memory.index result must be a pointer");
            (ptr.inner, ptr.address_space)
        };
        let elem_ty = cube_type_to_llvm(ctx, elem_ty);

        // A narrow binding is addressed by a 32-bit byte offset from its base, which a target
        // can keep in scalar registers: AMDGPU loads such an access as `saddr` plus an unsigned
        // 32-bit offset, where an element index widened before it is scaled takes a 64-bit
        // address per access.
        let index_bits = index
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(|int| int.width());
        let shift = (narrow_binding(ctx, address_space) && index_bits == Some(32))
            .then(|| byte_shift(ctx, elem_ty))
            .flatten();
        let (index, step_ty) = match shift {
            Some(shift) => (
                byte_offset(ctx, rewriter, index, shift),
                IntegerType::get(ctx, 8, Signedness::Signless).into(),
            ),
            None => (widen_gep_index(ctx, rewriter, index), elem_ty),
        };

        // An index stays inside the object it indexes (checked modes clamp it first), and it
        // is unsigned, so the offset never wraps below the base. Without these flags LLVM
        // cannot reassociate the address arithmetic or fold constant offsets into the
        // addressing mode of the access.
        let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
            ctx,
            base,
            vec![llvm::GepIndex::Value(index)],
            step_ty,
            GepNoWrapFlags::INBOUNDS | GepNoWrapFlags::NUW,
        );
        rewriter.insert_op(ctx, &gep);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![gep.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for StructFieldOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = self.base(ctx);
        let index = self.field(ctx).0;
        let struct_ty = operands_info
            .lookup_most_recent_type(base)
            .expect("Should have type")
            .unwrap_ptr(ctx);
        let padded_index = padded_field_index(ctx, struct_ty, index);

        let base_ty = cube_type_to_llvm(ctx, struct_ty);

        // A struct index always stays in bounds
        let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
            ctx,
            base,
            vec![
                llvm::GepIndex::Constant(0),
                llvm::GepIndex::Constant(padded_index as u32),
            ],
            base_ty,
            GepNoWrapFlags::INBOUNDS | GepNoWrapFlags::NUW,
        );
        rewriter.insert_op(ctx, &gep);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![gep.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for LoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let result = self.get_result(ctx);
        let res_cube_ty = operands_info
            .lookup_most_recent_type(result)
            .unwrap_or_else(|| result.get_type(ctx));
        let align = type_alignment(ctx, res_cube_ty);
        let res_ty = cube_type_to_llvm(ctx, res_cube_ty);

        let load = llvm::LoadOp::new(ctx, ptr, res_ty);
        load.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &load);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![load.get_result(ctx)],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for StoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        let ptr = self.ptr(ctx);
        let value_cube_ty = operands_info
            .lookup_most_recent_type(value)
            .unwrap_or_else(|| value.get_type(ctx));
        let align = type_alignment(ctx, value_cube_ty);

        let store = llvm::StoreOp::new(ctx, value, ptr);
        store.set_alignment(ctx, align);
        rewriter.insert_op(ctx, &store);
        rewriter.replace_operation(ctx, self.get_operation(), store.get_operation());
        Ok(())
    }
}

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{dialect::atomic, prelude::*};
use pliron::builtin::types::IntegerType;
use pliron_spirv::{
    ops,
    types::{FloatType, PointerType, VectorType},
};
use rspirv::spirv::{Capability, MemorySemantics, Scope, StorageClass};

use crate::{
    CustomCapabilitiesOp, lower::LowerOp, ops::to_spirv_dialect::ToSpirvDialectOp,
    types::ty_to_spirv_dialect,
};

define_scalar!(T);
define_size!(S);

macro_rules! float_atomic_capability {
    ($ty: ty, $op: ident) => {
        paste::paste! {
            #[op_interface_impl]
            impl CustomCapabilitiesOp for $ty {
                fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
                    let ty = self.result_type(ctx).deref(ctx);
                    let mut caps = vec![if let Some(float) = ty.downcast_ref::<FloatType>() {
                        match float.width {
                            64 => Capability::[<AtomicFloat64 $op EXT>],
                            32 => Capability::[<AtomicFloat32 $op EXT>],
                            16 => Capability::[<AtomicFloat16 $op EXT>],
                            _ => unreachable!(),
                        }
                    } else if ty.is::<VectorType>() {
                        Capability::AtomicFloat16VectorNV
                    } else {
                        unreachable!()
                    }];
                    if matches!(self.get_attr_memory(ctx).0, Scope::Device) {
                        caps.push(Capability::VulkanMemoryModelDeviceScopeKHR);
                    }
                    caps
                }
            }
        }
    };
}
float_atomic_capability!(ops::ext::AtomicFAddOp, Add);
float_atomic_capability!(ops::ext::AtomicFMinOp, MinMax);
float_atomic_capability!(ops::ext::AtomicFMaxOp, MinMax);

// These can't be detected automatically, because they're required from the intersection of memory
// model and scope. The auto-generated code can't see that both of those are set at once.
macro_rules! atomic_capability {
    ($($ty: ty),*) => {
        $(
            #[op_interface_impl]
            impl CustomCapabilitiesOp for $ty {
                fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
                    let mut caps = if matches!(self.get_attr_memory(ctx).0, Scope::Device) {
                        vec![Capability::VulkanMemoryModelDeviceScopeKHR]
                    } else {
                        vec![]
                    };
                    let ty = self.result_type(ctx).deref(ctx);
                    if let Some(int) = ty.downcast_ref::<IntegerType>()
                        && int.width() == 64
                    {
                        caps.push(Capability::Int64Atomics);
                    }
                    caps
                }
            }
        )*
    };
}

macro_rules! atomic_binop_to_spirv_dialect {
    ($ty: ty => $new_ty: ty $(,$extra:expr)*) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let op = self.get_operation();
                let ptr = op.operand(ctx, 0);
                let value = op.operand(ctx, 1);
                let scope = ptr_scope(ctx, ptr);
                let semantics = semantics_rw(ctx, ptr);
                let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, ptr, scope, semantics, value, $($extra),*);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}

atomic_capability!(ops::AtomicIAddOp, ops::AtomicISubOp);
atomic_capability!(ops::AtomicSMinOp, ops::AtomicUMinOp);
atomic_capability!(ops::AtomicSMaxOp, ops::AtomicUMaxOp);
atomic_capability!(ops::AtomicAndOp, ops::AtomicOrOp, ops::AtomicXorOp);
atomic_capability!(ops::AtomicLoadOp, ops::AtomicCompareExchangeOp);

#[op_interface_impl]
impl CustomCapabilitiesOp for ops::AtomicStoreOp {
    fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
        let mut caps = if matches!(self.get_attr_memory(ctx).0, Scope::Device) {
            vec![Capability::VulkanMemoryModelDeviceScopeKHR]
        } else {
            vec![]
        };
        let ty = self.get_operand_value(ctx).get_type(ctx).deref(ctx);
        if let Some(int) = ty.downcast_ref::<IntegerType>()
            && int.width() == 64
        {
            caps.push(Capability::Int64Atomics);
        }
        caps
    }
}

#[op_interface_impl]
impl CustomCapabilitiesOp for ops::AtomicExchangeOp {
    fn custom_capabilities(&self, ctx: &Context) -> Vec<Capability> {
        let mut caps = vec![];
        // Floats need no capability interestingly enough, but vectors do
        let ty = self.result_type(ctx).deref(ctx);
        if let Some(vector) = ty.downcast_ref::<VectorType>()
            && vector.element_type.deref(ctx).is::<FloatType>()
        {
            caps.push(Capability::AtomicFloat16VectorNV);
        } else if let Some(int) = ty.downcast_ref::<IntegerType>()
            && int.width() == 64
        {
            caps.push(Capability::Int64Atomics);
        }
        if matches!(self.get_attr_memory(ctx).0, Scope::Device) {
            caps.push(Capability::VulkanMemoryModelDeviceScopeKHR);
        }
        caps
    }
}

atomic_binop_to_spirv_dialect!(atomic::AtomicExchangeOp => ops::AtomicExchangeOp);

atomic_binop_to_spirv_dialect!(atomic::AtomicIAddOp => ops::AtomicIAddOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicFAddOp => ops::ext::AtomicFAddOp);

atomic_binop_to_spirv_dialect!(atomic::AtomicISubOp => ops::AtomicISubOp);

atomic_binop_to_spirv_dialect!(atomic::AtomicSMinOp => ops::AtomicSMinOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicUMinOp => ops::AtomicUMinOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicFMinOp => ops::ext::AtomicFMinOp);

atomic_binop_to_spirv_dialect!(atomic::AtomicSMaxOp => ops::AtomicSMaxOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicUMaxOp => ops::AtomicUMaxOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicFMaxOp => ops::ext::AtomicFMaxOp);

atomic_binop_to_spirv_dialect!(atomic::AtomicAndOp => ops::AtomicAndOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicOrOp => ops::AtomicOrOp);
atomic_binop_to_spirv_dialect!(atomic::AtomicXorOp => ops::AtomicXorOp);

#[op_interface_impl]
impl ToSpirvDialectOp for atomic::AtomicLoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let ptr = self.ptr(ctx);
        let scope = ptr_scope(ctx, ptr);
        let semantics = semantics_r(ctx, ptr);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op = ops::AtomicLoadOp::new(ctx, out_ty, ptr, scope, semantics);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for atomic::AtomicStoreOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let scope = ptr_scope(ctx, ptr);
        let semantics = semantics_w(ctx, ptr);
        let new_op = ops::AtomicStoreOp::new(ctx, ptr, scope, semantics, value);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for atomic::AtomicCompareExchangeWeakOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let cmp = self.cmp(ctx);
        let scope = ptr_scope(ctx, ptr);
        let semantics_succ = semantics_rw(ctx, ptr);
        let semantics_fail = semantics_r(ctx, ptr);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op = ops::AtomicCompareExchangeOp::new(
            ctx,
            out_ty,
            ptr,
            scope,
            semantics_succ,
            semantics_fail,
            value,
            cmp,
        );
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl LowerOp for atomic::AtomicFSubOp {
    fn lower(&self, scope: &cubecl_ir::Scope) -> Vec<Value> {
        let ptr = self.ptr(scope.ctx());
        let value = self.value(scope.ctx());
        scope.register_value_type::<T, S>(value);
        vec![atomic_f_sub::expand::<T, S>(scope, ptr.into(), value.into()).read_value(scope)]
    }
}

#[cube]
fn atomic_f_sub<T: Float, N: Size>(ptr: Atomic<Vector<T, N>>, value: Vector<T, N>) -> Vector<T, N> {
    ptr.fetch_add(-value)
}

pub fn semantics_r(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::ACQUIRE
}

pub fn semantics_w(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::RELEASE
}

pub fn semantics_rw(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::ACQUIRE_RELEASE
}

fn semantics_of(ctx: &Context, value: Value) -> MemorySemantics {
    match ptr_scope(ctx, value) {
        Scope::Device => MemorySemantics::UNIFORM_MEMORY,
        Scope::Workgroup => MemorySemantics::WORKGROUP_MEMORY,
        Scope::Subgroup => MemorySemantics::SUBGROUP_MEMORY,
        other => unreachable!("Invalid scope for atomic operation, {other:?}"),
    }
}

fn ptr_scope(ctx: &Context, value: Value) -> Scope {
    let ty = value.get_type(ctx).deref(ctx);
    if let Some(ptr_ty) = ty.downcast_ref::<PointerType>() {
        match ptr_ty.storage_class {
            StorageClass::StorageBuffer
            | StorageClass::PhysicalStorageBuffer
            | StorageClass::Uniform => Scope::Device,
            StorageClass::Workgroup => Scope::Workgroup,
            StorageClass::Function => Scope::Invocation,
            other => unreachable!("Invalid scope for atomic operation, {other:?}"),
        }
    } else {
        panic!("Should be ptr")
    }
}

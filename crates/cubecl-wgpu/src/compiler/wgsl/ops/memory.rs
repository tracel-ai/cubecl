use cubecl_ir::{
    AddressSpace, CanMaterialize, GlobalState, NoSideEffects, Scope,
    attributes::ZeroAttr,
    dialect::{general::PoisonOp, memory::*},
    ident,
    interfaces::memory_slot::{
        DestructurableConstructorOpInterface, DestructurableTypeInterface, DestructurableValueSlot,
        ValueSlot,
    },
    prelude::*,
    try_cast_ty,
    types::PointerType,
};
use pliron::{
    arg_err,
    attribute::{AttrObj, boxed_attr_cast},
    builtin::ops::ConstantOp,
    opts::mem2reg::{AllocInfo, PromotableAllocationInterface},
    utils::table::{HMap, SmallSet},
};

use crate::compiler::wgsl::{
    AddressOfOp, GlobalVariableOp,
    lower::LowerOp,
    ops::general::attr_to_wgsl,
    to_wgsl::{TypeExtWgsl, wasm_inventory_root, wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

wasm_inventory_root!(DeclareLocalOp);

#[cube_op(name = "wgsl.declare_local")]
#[result_ty(argument)]
#[op_traits(NoSideEffects, CanMaterialize)]
pub struct DeclareLocalOp {
    pub value_ty: TypeAttr,
    #[attribute(optional, untyped)]
    pub initializer: AttrObj,
}

#[op_interface_impl]
impl PromotableAllocationInterface for DeclareLocalOp {
    fn alloc_info(&self, ctx: &Context) -> Vec<AllocInfo> {
        vec![AllocInfo {
            ptr: self.get_result(ctx),
            ty: self.value_ty(ctx).get_type(ctx),
        }]
    }

    fn default_value(
        &self,
        ctx: &mut Context,
        inserter: &mut dyn Inserter,
        alloc_info: &AllocInfo,
    ) -> cubecl_core::ir::prelude::Result<Value> {
        if alloc_info.ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        if let Some(initializer) = self.initializer(ctx).map(|it| it.clone()) {
            let initializer = boxed_attr_cast(initializer).unwrap();
            let constant = ConstantOp::new(ctx, initializer);
            inserter.insert_op(ctx, &constant);
            Ok(constant.get_result(ctx))
        } else {
            let poison = PoisonOp::new(ctx, alloc_info.ty);
            inserter.insert_op(ctx, &poison);
            Ok(poison.get_result(ctx))
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        rewriter: &mut dyn Rewriter,
        alloc_infos: &[AllocInfo],
    ) -> cubecl_core::ir::prelude::Result<()> {
        if alloc_infos.len() != 1 || alloc_infos[0].ptr != self.get_result(ctx) {
            return arg_err!(self.loc(ctx), UnrelatedAllocInfo);
        }
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl DestructurableConstructorOpInterface for DeclareLocalOp {
    fn destructurable_values(&self, ctx: &Context) -> Vec<DestructurableValueSlot> {
        if let Some(init) = self.initializer(ctx)
            && !init.is::<ZeroAttr>()
        {
            return vec![];
        }
        let value_ty = self.value_ty(ctx).get_type(ctx);
        let ty = value_ty.deref(ctx);
        let Some(destructurable) = type_cast::<dyn DestructurableTypeInterface>(&*ty) else {
            return vec![];
        };
        let Some(destructured_type) = destructurable.subelement_index_map(ctx) else {
            return vec![];
        };

        vec![DestructurableValueSlot {
            slot: ValueSlot::new(self.get_result(ctx), value_ty),
            subelement_types: destructured_type,
        }]
    }

    fn destructure(
        &self,
        ctx: &mut Context,
        _value: &DestructurableValueSlot,
        used_indices: &SmallSet<AttrObj, 8>,
        rewriter: &mut PassRewriter,
        new_constructors: &mut Vec<TraitOp<dyn DestructurableConstructorOpInterface>>,
    ) -> HMap<AttrObj, ValueSlot> {
        let init = self.initializer(ctx).map(|it| {
            assert!(it.is::<ZeroAttr>());
        });
        let destructured_type = {
            let ty = self.value_ty(ctx).get_type(ctx).deref(ctx);
            let destructurable = try_cast_ty!(ty, ctx, dyn DestructurableTypeInterface);
            destructurable.subelement_index_map(ctx).unwrap()
        };

        let mut slot_map = HMap::new();
        for used_index in used_indices {
            let value_ty = destructured_type[used_index];
            let init = init.map(|_| ZeroAttr::new(value_ty).into());
            let result_ty = PointerType::get(ctx, value_ty, AddressSpace::Local).to_handle();
            let suballoc = DeclareLocalOp::new(ctx, result_ty, value_ty, init);
            rewriter.append_op(ctx, &suballoc);

            let slot = ValueSlot::new(suballoc.get_result(ctx), value_ty);
            slot_map.insert(used_index.clone(), slot);
            new_constructors.push(TraitOp::try_from_op(suballoc.get_operation(), ctx).unwrap());
        }

        slot_map
    }

    fn handle_destructuring_complete(
        &self,
        ctx: &mut Context,
        value: &DestructurableValueSlot,
        rewriter: &mut PassRewriter,
    ) -> Option<TraitOp<dyn DestructurableConstructorOpInterface>> {
        assert_eq!(value.slot.value, self.get_result(ctx));
        rewriter.erase_operation(ctx, self.get_operation());
        None
    }
}

wgsl_op!(DeclareLocalOp, |op, ctx| {
    let value_ty = op.value_ty(ctx).get_type(ctx).to_wgsl(ctx);
    let name = op.get_result(ctx).name(ctx);
    let init = op.initializer(ctx).map(|init| attr_to_wgsl(ctx, &**init));
    let init = init.map(|init| format!("= {init}")).unwrap_or_default();
    format!("var<function> {name}_store: {value_ty}{init};\nlet {name} = &{name}_store;\n")
});

#[op_interface_impl]
impl LowerOp for DeclareVariableOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx_mut();
        let addr_space = self.addr_space(ctx).0;
        let ty = self.result_type(ctx);
        let value_ty = self.value_ty(ctx).get_type(ctx);
        match addr_space {
            AddressSpace::Global(_) => panic!("Global vars not supported"),
            AddressSpace::Shared => {
                let name = ident(format!("shared_{}", self.get_result(ctx).name(ctx)));
                let var = GlobalVariableOp::new(ctx, value_ty, addr_space, None, None);
                let entry = ctx.aux_ty::<GlobalState>().entry_func.get_operation();
                var.set_symbol_name(ctx, name.clone());
                var.get_operation().insert_before(ctx, entry);
                vec![scope.register_with_result(&AddressOfOp::new(ctx, ty, name))]
            }
            AddressSpace::Local => {
                let init = self.initializer(ctx).map(|it| it.clone());
                let local = DeclareLocalOp::new(ctx, ty, value_ty, init);
                vec![scope.register_with_result(&local)]
            }
        }
    }
}

wgsl_op_with_out!(IndexOp; |op, ctx| {
    format!("&(*{})[{}]", op.base(ctx).name(ctx), op.index(ctx).name(ctx))
});
wgsl_op_with_out!(StructFieldOp; |op, ctx| {
    format!("&(*{})._{}", op.base(ctx).name(ctx), op.field(ctx).0)
});

wgsl_op_with_out!(LoadOp; |op, ctx| format!("*{}", op.ptr(ctx).name(ctx)));
wgsl_op!(StoreOp, |op, ctx| {
    let ptr = op.ptr(ctx).name(ctx);
    format!("*{ptr} = {};\n", op.value(ctx).name(ctx))
});
wgsl_op!(CopyOp, |op, ctx| {
    assert_eq!(op.len(ctx).0, 1, "WGSL doesn't support bulk copy");
    let dest = op.destination(ctx).name(ctx);
    format!("*{dest} = *{};\n", op.source(ctx).name(ctx))
});

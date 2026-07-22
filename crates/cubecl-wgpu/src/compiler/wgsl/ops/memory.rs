use cubecl_ir::{
    AddressSpace, CanMaterialize, NoSideEffects, Scope, dialect::memory::*, ident, prelude::*,
};
use pliron::attribute::AttrObj;

use crate::compiler::wgsl::{
    AddressOfOp, GlobalVariableOp,
    lower::LowerOp,
    ops::general::attr_to_wgsl,
    to_wgsl::{TypeExtWgsl, wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

#[cube_op(name = "wgsl.declare_local")]
#[result_ty(argument)]
#[op_traits(NoSideEffects, CanMaterialize)]
pub struct DeclareLocalOp {
    pub value_ty: TypeAttr,
    #[attribute(optional, untyped)]
    pub initializer: AttrObj,
}

wgsl_op!(DeclareLocalOp, |op, ctx| {
    let value_ty = op.value_ty(ctx).get_type(ctx).to_wgsl(ctx);
    let name = op.get_result(ctx).name(ctx);
    let init = op.initializer(ctx).map(|init| attr_to_wgsl(ctx, &**init));
    let init = init.map(|init| format!("= {init}")).unwrap_or_default();
    format!("var<function> {name}_store: {value_ty}{init};\nlet {name} = &{name}_store;")
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
                let module = self.get_operation().parent_module(ctx).get_body(ctx, 0);
                let var = GlobalVariableOp::new(ctx, value_ty, addr_space);
                var.set_symbol_name(ctx, name.clone());
                var.get_operation().insert_at_front(module, ctx);
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

wgsl_op_with_out!(IndexOp, |op, ctx| {
    format!("&{}[{}]", op.base(ctx).name(ctx), op.index(ctx).name(ctx))
});

wgsl_op_with_out!(LoadOp, |op, ctx| format!("*{}", op.ptr(ctx).name(ctx)));
wgsl_op!(StoreOp, |op, ctx| {
    format!("*{} = {};", op.ptr(ctx).name(ctx), op.value(ctx).name(ctx))
});
wgsl_op!(CopyOp, |op, ctx| {
    assert_eq!(op.len(ctx).0, 1, "WGSL doesn't support bulk copy");
    let dest = op.destination(ctx).name(ctx);
    format!("*{dest} = *{};", op.source(ctx).name(ctx))
});

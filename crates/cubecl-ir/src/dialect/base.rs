use crate::{
    prelude::*,
    types::{AtomicType, PointerType},
};

#[allow(clippy::wrong_self_convention)]
pub trait OperationPtrExt: Sized {
    fn is_op<T: Op>(self, ctx: &Context) -> bool {
        self.as_op::<T>(ctx).is_some()
    }
    fn is_terminator(self, ctx: &Context) -> bool {
        self.impls::<dyn IsTerminatorInterface>(ctx)
    }
    fn impls<T: ?Sized + OpInterfaceMarker + 'static>(self, ctx: &Context) -> bool {
        op_impls::<T>(&*self.dyn_op(ctx))
    }

    fn as_op<T: Op>(self, ctx: &Context) -> Option<T>;
    fn dyn_op(self, ctx: &Context) -> OpObj;
    fn operand(self, ctx: &Context, idx: usize) -> Value;
    fn operand_as_use(self, ctx: &Context, idx: usize) -> Use<Value>;
    fn operands(self, ctx: &Context) -> Vec<Value>;
    fn operands_as_uses(self, ctx: &Context) -> Vec<Use<Value>>;
    fn result(self, ctx: &Context) -> Value;
    fn results(self, ctx: &Context) -> Vec<Value>;
    fn result_types(self, ctx: &Context) -> Vec<TypeHandle>;
    fn regions(self, ctx: &Context) -> Vec<Ptr<Region>>;
    fn result_names(self, ctx: &Context) -> Vec<Option<Identifier>>;
    fn opt_result(self, ctx: &Context) -> Option<Value>;
    fn set_attr<T: Attribute>(self, ctx: &Context, key: &Identifier, value: T);
    fn parent_module(self, ctx: &Context) -> ModuleOp;

    fn is_ancestor_of(self, ctx: &Context, other: Ptr<Operation>) -> bool;
}

pub trait BlockPtrExt: Sized {
    fn arguments(self, ctx: &Context) -> Vec<Value>;
    fn is_entry_block(&self, ctx: &Context) -> bool;
    fn ops_with_interface<T: OpInterfaceMarker + ?Sized + 'static>(
        &self,
        ctx: &Context,
    ) -> impl Iterator<Item = TraitOp<T>>;
}

pub trait RegionPtrExt: Sized {
    fn arguments(self, ctx: &Context) -> Vec<Value>;
    fn is_empty(&self, ctx: &Context) -> bool;
}

impl OperationPtrExt for Ptr<Operation> {
    fn as_op<T: Op>(self, ctx: &Context) -> Option<T> {
        Operation::get_op(self, ctx)
    }
    fn dyn_op(self, ctx: &Context) -> OpObj {
        Operation::get_op_dyn(self, ctx)
    }
    fn operand(self, ctx: &Context, idx: usize) -> Value {
        Operation::get_operand(&self.deref(ctx), idx)
    }
    fn operand_as_use(self, ctx: &Context, idx: usize) -> Use<Value> {
        Operation::get_operand_as_use(&self.deref(ctx), idx)
    }
    fn operands(self, ctx: &Context) -> Vec<Value> {
        self.deref(ctx).operands().collect()
    }
    fn operands_as_uses(self, ctx: &Context) -> Vec<Use<Value>> {
        self.deref(ctx).operands_as_uses().collect()
    }
    fn result(self, ctx: &Context) -> Value {
        Operation::get_result(&self.deref(ctx), 0)
    }
    fn results(self, ctx: &Context) -> Vec<Value> {
        self.deref(ctx).results().collect()
    }
    fn result_types(self, ctx: &Context) -> Vec<TypeHandle> {
        self.deref(ctx).result_types().collect()
    }
    fn regions(self, ctx: &Context) -> Vec<Ptr<Region>> {
        self.deref(ctx).regions().collect()
    }
    fn result_names(self, ctx: &Context) -> Vec<Option<Identifier>> {
        self.deref(ctx)
            .results()
            .map(|it| it.given_name(ctx))
            .collect()
    }
    fn opt_result(self, ctx: &Context) -> Option<Value> {
        self.deref(ctx).results().next()
    }
    fn set_attr<T: Attribute>(self, ctx: &Context, key: &Identifier, value: T) {
        self.deref_mut(ctx).attributes.set(key.clone(), value);
    }
    fn parent_module(self, ctx: &Context) -> ModuleOp {
        if self.is_op::<ModuleOp>(ctx) {
            return self.as_op(ctx).unwrap();
        }
        let mut op = self;
        while let Some(parent) = op.deref(ctx).get_parent_op(ctx) {
            if parent.is_op::<ModuleOp>(ctx) {
                return parent.as_op(ctx).unwrap();
            }
            op = parent;
        }
        panic!("Op is not contained in any module")
    }

    fn is_ancestor_of(self, ctx: &Context, mut child: Ptr<Operation>) -> bool {
        while let Some(parent) = child.deref(ctx).get_parent_op(ctx) {
            if parent == self {
                return true;
            }
            child = parent;
        }
        false
    }
}

impl BlockPtrExt for Ptr<BasicBlock> {
    fn arguments(self, ctx: &Context) -> Vec<Value> {
        self.deref(ctx).arguments().collect()
    }

    fn is_entry_block(&self, ctx: &Context) -> bool {
        self.deref(ctx)
            .get_parent_region()
            .is_some_and(|region| region.deref(ctx).get_entry_block() == Some(*self))
    }

    fn ops_with_interface<T: OpInterfaceMarker + ?Sized + 'static>(
        &self,
        ctx: &Context,
    ) -> impl Iterator<Item = TraitOp<T>> {
        self.deref(ctx)
            .iter(ctx)
            .filter_map(|op| TraitOp::try_from_op(op, ctx))
    }
}

impl RegionPtrExt for Ptr<Region> {
    fn arguments(self, ctx: &Context) -> Vec<Value> {
        match self.entry_node(ctx) {
            Some(entry) => entry.arguments(ctx),
            None => vec![],
        }
    }

    fn is_empty(&self, ctx: &Context) -> bool {
        self.entry_node(ctx).is_none()
    }
}

macro_rules! pure_unop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[$crate::prelude::op_interfaces(
            SameOperandsType,
            SameOperandsAndResultType,
            $crate::interfaces::TriviallyUnrollable
        )]
        #[$crate::prelude::op_traits(
            $crate::CanMaterialize,
            $crate::Pure,
            $crate::PropagatesUniformity
        )]
        pub struct $ty {
            pub input: Value,
        }
    };
}
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    common_traits::Named,
    graph::ControlFlowGraph,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::{OpInterfaceMarker, OpObj, op_impls},
    region::Region,
    r#type::TypeHandle,
    value::Use,
};
pub(crate) use pure_unop;

macro_rules! pure_binop {
    ($name: literal, $ty: ident) => {
        #[cubecl_macros_internal::cube_op(name = $name)]
        #[result_ty(same_as = lhs)]
        #[$crate::prelude::op_interfaces(
            SameOperandsType,
            SameOperandsAndResultType,
            $crate::interfaces::TriviallyUnrollable
        )]
        #[$crate::prelude::op_traits(
            $crate::CanMaterialize,
            $crate::Pure,
            $crate::PropagatesUniformity
        )]
        pub struct $ty {
            pub lhs: Value,
            pub rhs: Value,
        }
    };
}
pub(crate) use pure_binop;

pub fn ptr_value_ty(ctx: &Context, input: &Value) -> TypeHandle {
    let in_ty = input.get_type(ctx).deref(ctx);
    let ptr_ty = in_ty.downcast_ref::<PointerType>();
    let mut inner = ptr_ty.expect("Should be a pointer").inner;
    // If type is atomic, it's the value inside it after load
    {
        let inner_ref = inner.deref(ctx);
        if let Some(AtomicType {
            inner: inner_val, ..
        }) = inner_ref.downcast_ref()
        {
            inner = *inner_val;
        }
    }
    inner
}

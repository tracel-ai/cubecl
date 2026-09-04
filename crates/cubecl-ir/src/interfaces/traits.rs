use pliron::{
    builtin::ops::{FuncOp, ModuleOp},
    dict_key,
};

#[macro_export]
macro_rules! Pure {
    ($ty: ty) => {
        $crate::NoSideEffects!($ty);
        $crate::NoMemoryEffect!($ty);
    };
}

#[macro_export]
macro_rules! CanMaterialize {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::MaterializableOp for $ty {
            fn materialize(
                &self,
                ctx: &mut pliron::context::Context,
                result_ty: Vec<pliron::r#type::TypeHandle>,
                operands: Vec<Value>,
                attributes: pliron::attribute::AttributeDict,
            ) -> pliron::context::Ptr<pliron::operation::Operation> {
                use pliron::op::Op;
                let op = pliron::operation::Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    result_ty,
                    operands,
                    vec![],
                    0,
                );
                op.deref_mut(ctx).attributes = attributes;
                op
            }
        }
    };
}

#[macro_export]
macro_rules! NoMemoryEffect {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::MemoryEffects for $ty {
            fn memory_effects(
                &self,
                _ctx: &pliron::context::Context,
            ) -> $crate::alloc::vec::Vec<$crate::interfaces::MemoryEffect> {
                $crate::alloc::vec![]
            }
        }
    };
}

#[macro_export]
macro_rules! PropagatesUniformity {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::uniformity::UniformOpInterface for $ty {
            fn uniformity(
                &self,
                _ctx: &pliron::context::Context,
                operands: &[$crate::interfaces::uniformity::Uniformity],
            ) -> $crate::interfaces::uniformity::Uniformity {
                operands
                    .iter()
                    .copied()
                    .min()
                    .unwrap_or($crate::interfaces::uniformity::Uniformity::Device)
            }
        }
    };
}

#[macro_export]
macro_rules! ReturnLike {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::control_flow::RegionBranchTerminatorOpInterface for $ty {
            fn successor_operands(&self, ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
                self.get_operation().deref(ctx).operands().collect()
            }
        }

        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::uniformity::UniformRegionTerminatorOpInterface for $ty {
            fn successor_region_uniformity(
                &self,
                ctx: &Context,
                _operands: &[$crate::interfaces::uniformity::Uniformity],
            ) -> Vec<$crate::interfaces::uniformity::Uniformity> {
                self.all_successor_regions(ctx)
                    .iter()
                    .map(|_| $crate::interfaces::uniformity::Uniformity::Cube)
                    .collect()
            }
        }

        #[::pliron::derive::op_interface_impl]
        impl $crate::interfaces::ReturnLike for $ty {}
    };
}

#[macro_export]
macro_rules! HasSideEffects {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl pliron::opts::dce::SideEffects for $ty {
            fn has_side_effects(&self, _ctx: &pliron::context::Context) -> bool {
                true
            }
        }
    };
}

#[macro_export]
macro_rules! NoSideEffects {
    ($ty: ty) => {
        #[::pliron::derive::op_interface_impl]
        impl pliron::opts::dce::SideEffects for $ty {
            fn has_side_effects(&self, _ctx: &pliron::context::Context) -> bool {
                false
            }
        }
    };
}

dict_key!(
    /// Key for symbol visibility attribute when the operation defines a symbol visibility.
    ATTR_KEY_SYM_VISIBILITY,
    "sym_visibility"
);

#[macro_export]
macro_rules! SymbolVisibility {
    ($ty: ty) => {
        #[$crate::pliron::derive::op_interface_impl]
        impl $crate::interfaces::control_flow::SymbolOpInterface for $ty {
            fn get_visibility(
                &self,
                ctx: &$crate::pliron::context::Context,
            ) -> $crate::interfaces::control_flow::SymbolVisiblity {
                use $crate::pliron::op::Op;
                let op = self.get_operation().deref(ctx);
                let attr: Option<&$crate::attributes::SymbolVisibilityAttr> = op
                    .attributes
                    .get(&$crate::interfaces::traits::ATTR_KEY_SYM_VISIBILITY);
                attr.map(|it| it.0)
                    .unwrap_or($crate::interfaces::control_flow::SymbolVisiblity::Public)
            }

            fn set_visibility(
                &self,
                ctx: &mut $crate::pliron::context::Context,
                visibility: $crate::interfaces::control_flow::SymbolVisiblity,
            ) {
                use $crate::pliron::op::Op;
                let mut op = self.get_operation().deref_mut(ctx);
                op.attributes.set(
                    $crate::interfaces::traits::ATTR_KEY_SYM_VISIBILITY.clone(),
                    $crate::attributes::SymbolVisibilityAttr(visibility),
                );
            }
        }
    };
}

SymbolVisibility!(ModuleOp);
SymbolVisibility!(FuncOp);

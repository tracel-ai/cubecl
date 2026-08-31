#[macro_export]
macro_rules! ReturnLike {
    ($ty: ty) => {
        #[op_interface_impl]
        impl $crate::interfaces::control_flow::RegionBranchTerminatorOpInterface for $ty {
            fn successor_operands(&self, ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
                self.get_operation().deref(ctx).operands().collect()
            }
        }

        #[op_interface_impl]
        impl $crate::interfaces::ReturnLike for $ty {}
    };
}

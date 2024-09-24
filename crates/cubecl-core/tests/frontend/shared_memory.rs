use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn shared_memory_read_write<T: Numeric>(#[comptime] sm_size: u32) {
    let mut shared = SharedMemory::<T>::new(sm_size);
    shared[0] = T::from_int(3);
    let _ = shared[0];
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = f32;

    #[test]
    fn cube_support_shared_memory() {
        let mut context = CubeContext::default();

        shared_memory_read_write::expand::<ElemType>(&mut context, 512);
        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            inline_macro_ref()
        );
    }

    fn inline_macro_ref() -> String {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let var = scope.create_local(item);
        let pos: Variable = 0u32.into();

        // Create
        let shared = scope.create_shared(item, 512);

        // Write
        cpa!(scope, shared[pos] = 3.0_f32);

        // Read
        cpa!(scope, var = shared[pos]);

        format!("{:?}", scope.operations)
    }
}

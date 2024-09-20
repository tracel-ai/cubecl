use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, comptime};

#[derive(Clone)]
pub struct State {
    cond: bool,
    bound: u32,
}

impl Init for State {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[cube]
pub fn comptime_if_else<T: Numeric>(lhs: T, #[comptime] cond: bool) {
    if cond {
        let _ = lhs + T::from_int(4);
    } else {
        let _ = lhs - T::from_int(5);
    }
}

#[cube]
#[allow(clippy::collapsible_else_if)]
pub fn comptime_else_then_if<T: Numeric>(lhs: T, #[comptime] cond1: bool, #[comptime] cond2: bool) {
    if cond1 {
        let _ = lhs + T::from_int(4);
    } else {
        if cond2 {
            let _ = lhs + T::from_int(5);
        } else {
            let _ = lhs - T::from_int(6);
        }
    }
}

#[cube]
pub fn comptime_float() {
    let comptime_float = 0.0f32;
    let _runtime_float = comptime_float.runtime();
}

#[cube]
pub fn comptime_elsif<T: Numeric>(lhs: T, #[comptime] cond1: bool, #[comptime] cond2: bool) {
    if cond1 {
        let _ = lhs + T::from_int(4);
    } else if cond2 {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_elsif_with_runtime1<T: Numeric>(lhs: T, #[comptime] comptime_cond: bool) {
    let runtime_cond = lhs >= T::from_int(2);
    if comptime_cond {
        let _ = lhs + T::from_int(4);
    } else if runtime_cond {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_elsif_with_runtime2<T: Numeric>(lhs: T, #[comptime] comptime_cond: bool) {
    let runtime_cond = lhs >= T::from_int(2);
    if runtime_cond {
        let _ = lhs + T::from_int(4);
    } else if comptime_cond {
        let _ = lhs + T::from_int(5);
    } else {
        let _ = lhs - T::from_int(6);
    }
}

#[cube]
pub fn comptime_if_expr<T: Numeric>(lhs: T, #[comptime] x: u32, #[comptime] y: u32) {
    let y2 = x + y;

    if x < y2 {
        let _ = lhs + T::from_int(4);
    } else {
        let _ = lhs - T::from_int(5);
    }
}

#[cube]
pub fn comptime_with_map_bool<T: Numeric>(#[comptime] state: State) -> T {
    let cond = state.cond;

    let mut x = T::from_int(3);
    if cond {
        x += T::from_int(4);
    } else {
        x -= T::from_int(4);
    }
    x
}

#[cube]
pub fn comptime_with_map_uint<T: Numeric>(#[comptime] state: State) -> T {
    let bound = state.bound;

    let mut x = T::from_int(3);
    #[unroll]
    for _ in 0..bound {
        x += T::from_int(4);
    }

    x
}

fn rust_function(input: u32) -> u32 {
    input + 2
}

#[cube]
pub fn comptime_block<T: Numeric>(a: T) -> T {
    let comptime_val = comptime! { rust_function(2) as i64 };

    a + T::from_int(comptime_val)
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        frontend::{CubeContext, CubePrimitive},
        ir::{Elem, Item, Variable},
    };
    use pretty_assertions::assert_eq;

    type ElemType = f32;

    #[test]
    fn cube_comptime_if_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        comptime_if_else::expand::<ElemType>(&mut context, lhs.into(), true);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime(true)
        );
    }

    #[test]
    fn cube_comptime_if_numeric_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        comptime_if_expr::expand::<ElemType>(&mut context, lhs.into(), 4, 5);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime(true)
        );
    }

    #[test]
    fn cube_comptime_else_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        comptime_if_else::expand::<ElemType>(&mut context, lhs.into(), false);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime2(false)
        );
    }

    #[test]
    fn cube_comptime_elsif_test() {
        for cond1 in [false, true] {
            for cond2 in [false, true] {
                let mut context1 = CubeContext::default();
                let lhs = context1.create_local_binding(Item::new(ElemType::as_elem()));
                comptime_else_then_if::expand::<ElemType>(&mut context1, lhs.into(), cond1, cond2);
                let scope1 = context1.into_scope();

                let mut context2 = CubeContext::default();
                let lhs = context2.create_local_binding(Item::new(ElemType::as_elem()));
                comptime_elsif::expand::<ElemType>(&mut context2, lhs.into(), cond1, cond2);
                let scope2 = context2.into_scope();

                assert_eq!(
                    format!("{:?}", scope1.operations),
                    format!("{:?}", scope2.operations),
                );
            }
        }
    }

    #[test]
    fn cube_comptime_elsif_runtime1_test() {
        for cond in [false, true] {
            let mut context = CubeContext::default();
            let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));
            comptime_elsif_with_runtime1::expand::<ElemType>(&mut context, lhs.into(), cond);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:?}", scope.operations),
                inline_macro_ref_elsif_runtime1(cond)
            );
        }
    }

    #[test]
    fn cube_comptime_elsif_runtime2_test() {
        for cond in [false, true] {
            let mut context = CubeContext::default();
            let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

            comptime_elsif_with_runtime2::expand::<ElemType>(&mut context, lhs.into(), cond);
            let scope = context.into_scope();

            assert_eq!(
                format!("{:#?}", scope.operations),
                inline_macro_ref_elsif_runtime2(cond)
            );
        }
    }

    #[test]
    fn cube_comptime_map_bool_test() {
        let mut context1 = CubeContext::default();
        let mut context2 = CubeContext::default();

        let comptime_state_true = State {
            cond: true,
            bound: 4,
        };
        let comptime_state_false = State {
            cond: false,
            bound: 4,
        };

        comptime_with_map_bool::expand::<ElemType>(&mut context1, comptime_state_true);
        comptime_with_map_bool::expand::<ElemType>(&mut context2, comptime_state_false);

        let scope1 = context1.into_scope();
        let scope2 = context2.into_scope();

        assert_ne!(
            format!("{:?}", scope1.operations),
            format!("{:?}", scope2.operations)
        );
    }

    #[test]
    fn cube_comptime_map_uint_test() {
        let mut context = CubeContext::default();

        let comptime_state = State {
            cond: true,
            bound: 4,
        };

        comptime_with_map_uint::expand::<ElemType>(&mut context, comptime_state);

        let scope = context.into_scope();

        assert!(!format!("{:?}", scope.operations).contains("RangeLoop"));
    }

    #[test]
    fn cube_comptime_block_test() {
        let mut context = CubeContext::default();

        let a = context.create_local_binding(Item::new(ElemType::as_elem()));

        comptime_block::expand::<ElemType>(&mut context, a.into());

        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_comptime_block()
        );
    }

    fn inline_macro_ref_comptime(cond: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y = scope.create_local(item);

        if cond {
            cpa!(scope, y = x + 4.0f32);
        } else {
            cpa!(scope, y = x - 5.0f32);
        };

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_comptime2(cond: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();

        if cond {
            cpa!(scope, x = x + 4.0f32);
        } else {
            cpa!(scope, x = x - 5.0f32);
        };

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_elsif_runtime1(comptime_cond: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let runtime_cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        cpa!(scope, runtime_cond = x >= 2.0f32);

        if comptime_cond {
            cpa!(scope, y = x + 4.0f32);
        } else {
            cpa!(&mut scope, if(runtime_cond).then(|scope| {
                cpa!(scope, y = x + 5.0f32);
            }).else(|scope| {
                cpa!(scope, y = x - 6.0f32);
            }));
        };

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_elsif_runtime2(comptime_cond: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let runtime_cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        cpa!(scope, runtime_cond = x >= 2.0f32);

        cpa!(&mut scope, if(runtime_cond).then(|scope| {
            cpa!(scope, y = x + 4.0f32);
        }).else(|scope| {
            if comptime_cond {
                cpa!(scope, y = x + 5.0f32);
            } else {
                cpa!(scope, y = x - 6.0f32);
            }
        }));

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_comptime_block() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let a = context.create_local_variable(item);
        let comptime_var: Variable = ElemType::from_int(4).into();

        let mut scope = context.into_scope();
        let x: Variable = a.into();
        cpa!(scope, x = x + comptime_var);

        format!("{:?}", scope.operations)
    }
}

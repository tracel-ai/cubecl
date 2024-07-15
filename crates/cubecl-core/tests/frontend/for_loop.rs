use cubecl_core as cubecl;
use cubecl_core::{
    cube,
    frontend::branch::range,
    frontend::{Array, Comptime, CubeContext, CubePrimitive, Float, UInt, F32},
};

type ElemType = F32;

#[cube(debug)]
pub fn for_loop<F: Float>(mut lhs: Array<F>, rhs: F, end: UInt, unroll: Comptime<bool>) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    for i in range(0u32, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn for_loop<F: Float>(mut lhs: Array<F>, rhs: F, end: UInt, unroll: Comptime<bool>) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;
    for i in range(0u32, end, unroll) {
        lhs[i] = tmp2 + lhs[i];
    }
}
#[doc = "Module containing the expand function of for_loop."]
pub mod for_loop {
    use super::*;
    #[allow(unused_mut)]
    #[allow(clippy::too_many_arguments)]
    #[doc = r" Expanded Cube function"]
    pub fn __expand<F: Float>(
        context: &mut cubecl::frontend::CubeContext,
        mut lhs: <Array<F> as cubecl::frontend::CubeType>::ExpandType,
        rhs: <F as cubecl::frontend::CubeType>::ExpandType,
        end: <UInt as cubecl::frontend::CubeType>::ExpandType,
        unroll: <Comptime<bool> as cubecl::frontend::CubeType>::ExpandType,
    ) -> () {
        let tmp1 = {
            let _inner = {
                let _lhs = rhs.clone();
                let _rhs = rhs.clone();
                cubecl::frontend::mul::expand(context, _lhs, _rhs)
            };
            _inner.init(context)
        };
        let tmp2 = {
            let _inner = {
                let _lhs = tmp1;
                let _rhs = rhs;
                cubecl::frontend::add::expand(context, _lhs, _rhs)
            };
            _inner.init(context)
        };
        {
            let _start = 0u32;
            let _end = end;
            let _unroll = unroll;
            cubecl::frontend::branch::range_expand(context, _start, _end, _unroll, |context, i| {
                {
                    let _array = lhs.clone();
                    let _index = i.clone();
                    let _value = {
                        let _lhs = tmp2.clone();
                        let _rhs = {
                            let _array = lhs.clone();
                            let _index = i;
                            cubecl::frontend::index::expand(context, _array, _index)
                        };
                        cubecl::frontend::add::expand(context, _lhs, _rhs)
                    };
                    cubecl::frontend::index_assign::expand(context, _array, _index, _value)
                };
            });
        }
    }
}

mod tests {
    use cubecl::frontend::ExpandElement;
    use cubecl_core::{cpa, ir::Item};

    use super::*;

    #[test]
    fn test_for_loop_with_unroll() {
        let mut context = CubeContext::root();
        let unroll = true;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end: ExpandElement = 4u32.into();

        for_loop::__expand::<ElemType>(&mut context, lhs.into(), rhs.into(), end.into(), unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    #[test]
    fn test_for_loop_no_unroll() {
        let mut context = CubeContext::root();
        let unroll = false;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local(Item::new(ElemType::as_elem()));
        let end: ExpandElement = 4u32.into();

        for_loop::__expand::<ElemType>(&mut context, lhs.into(), rhs.into(), end.into(), unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref(unroll));
    }

    fn inline_macro_ref(unroll: bool) -> String {
        let context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let lhs = scope.create_local_array(item, 4u32);
        let rhs = scope.create_local(item);
        let end = 4u32;

        // Kernel
        let tmp1 = scope.create_local(item);
        cpa!(scope, tmp1 = rhs * rhs);
        cpa!(scope, tmp1 = tmp1 + rhs);

        cpa!(
            &mut scope,
            range(0u32, end, unroll).for_each(|i, scope| {
                cpa!(scope, rhs = lhs[i]);
                cpa!(scope, rhs = tmp1 + rhs);
                cpa!(scope, lhs[i] = rhs);
            })
        );

        format!("{:?}", scope.operations)
    }
}

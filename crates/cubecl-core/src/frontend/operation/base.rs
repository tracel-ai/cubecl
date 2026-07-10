use cubecl_ir::{
    ExpandValue, Scope,
    dialect::{general::CopyOp, memory::IndexOp},
    interfaces::TypedExt,
    pliron::{
        builtin::op_interfaces::OneResultInterface, context::Context, op::Op, r#type::Typed,
        value::Value,
    },
    types::VectorType,
};
use cubecl_macros::cube;

use crate::{self as cubecl, prelude::*};

pub(crate) fn normalize_same_vectorization<const N: usize>(
    scope: &Scope,
    mut vals: [Value; N],
) -> [Value; N] {
    let max_vector_size = {
        let ctx = scope.ctx();
        vals.iter().map(|it| it.vector_size(ctx)).max().unwrap_or(1)
    };
    for val in vals.iter_mut() {
        let vector_size = val.vector_size(scope.ctx());
        if vector_size == 1 && max_vector_size > 1 {
            let scalar_ty = val.scalar_ty(scope.ctx());
            let out_ty = VectorType::get(scope.ctx(), scalar_ty, max_vector_size);
            *val = cast_value(scope, *val, out_ty.into());
        } else if vector_size != max_vector_size {
            panic!("Invalid vector size mismatch, expected same size or scalar")
        }
    }
    vals
}

pub(crate) fn binary_expand<F, O>(
    scope: &Scope,
    lhs: ExpandValue,
    rhs: ExpandValue,
    func: F,
) -> ExpandValue
where
    F: Fn(&mut Context, Value, Value) -> O,
    O: Op + OneResultInterface,
{
    let [lhs, rhs] =
        normalize_same_vectorization(scope, [lhs.read_value(scope), rhs.read_value(scope)]);
    let op = func(scope.ctx_mut(), lhs, rhs);
    scope.register_with_result(&op).into()
}

pub(crate) fn index_expand(scope: &Scope, list: Value, index: Value, checked: bool) -> Value {
    let op = IndexOp::maybe_checked(scope.ctx_mut(), list, index, checked);
    scope.register_with_result(&op)
}

pub(crate) fn assign_binop_expand<T: NativeCubeType + CanReadValue>(
    scope: &Scope,
    lhs: &mut NativeExpand<T>,
    rhs: NativeExpand<T>,
    func: impl Fn(&Scope, ExpandValue, ExpandValue) -> ExpandValue,
) where
    NativeExpand<T>: DerefExpand<Target = NativeExpand<T>>,
{
    let lhs_val = lhs.__expand_deref_method(scope);
    let out = func(scope, lhs_val.into(), rhs.into());
    assign::expand_element(scope, out, lhs.expand);
}

pub fn unary_expand<F, O>(scope: &Scope, input: ExpandValue, func: F) -> ExpandValue
where
    F: Fn(&mut Context, Value) -> O,
    O: Op + OneResultInterface,
{
    let input = input.read_value(scope);
    let op = func(scope.ctx_mut(), input);
    scope.register_with_result(&op).into()
}

pub fn init_expand(scope: &Scope, input: ExpandValue, mutable: bool) -> ExpandValue {
    let input = input.read_value(scope);
    let ty = input.get_type(scope.ctx());

    if mutable {
        let out = scope.create_local_mut(ty);
        assign::expand_element(scope, input.into(), out.into());
        out.into()
    } else {
        let op = CopyOp::new(scope.ctx_mut(), input);
        scope.register_with_result(&op).into()
    }
}

pub trait DivCeil: Int + CubeType<ExpandType: DivCeilExpand<Self>> {
    fn div_ceil(self, divisor: Self) -> Self;

    fn __expand_div_ceil(
        scope: &Scope,
        a: NativeExpand<Self>,
        b: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        a.__expand_div_ceil_method(scope, b)
    }
}

pub trait DivCeilExpand<E: Int> {
    fn __expand_div_ceil_method(self, scope: &Scope, divisor: Self) -> Self;
}

impl<E: DivCeil> DivCeilExpand<E> for NativeExpand<E> {
    fn __expand_div_ceil_method(self, scope: &Scope, divisor: NativeExpand<E>) -> NativeExpand<E> {
        div_ceil::expand::<E>(scope, self, divisor)
    }
}

macro_rules! impl_div_ceil {
    ($($ty:ty),*) => {
        $(
            impl DivCeil for $ty {
                #[allow(clippy::manual_div_ceil)] // Need to define div_ceil to use div_ceil!
                fn div_ceil(self, divisor: Self) -> Self {
                    (self + divisor - 1) / divisor
                }
            }
        )*
    };
}

impl_div_ceil!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl<E: Int> NativeExpand<E> {
    pub fn __expand_is_multiple_of_method(
        self,
        scope: &Scope,
        factor: NativeExpand<E>,
    ) -> NativeExpand<bool> {
        let modulo = self.__expand_rem_method(scope, factor);
        modulo.__expand_eq_method(scope, &E::from_int(0).into())
    }
}

#[cube]
pub fn div_ceil<E: Int>(a: E, b: E) -> E {
    (a + b - E::new(1)) / b
}

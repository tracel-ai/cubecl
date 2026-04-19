use cubecl_ir::{
    Arithmetic, BinaryOperator, Comparison, ElemType, IndexOperator, Instruction, Operation,
    Operator, Scope, Type, UnaryOperator, Variable, VectorSize,
};
use cubecl_macros::cube;

use crate::{self as cubecl, prelude::*};

pub(crate) fn binary_expand<F, Op>(
    scope: &mut Scope,
    lhs: Variable,
    rhs: Variable,
    func: F,
) -> Variable
where
    F: Fn(BinaryOperator) -> Op,
    Op: Into<Operation>,
{
    let item_lhs = lhs.ty;
    let item_rhs = rhs.ty;

    let vector_size = find_vectorization(item_lhs, item_rhs);

    let item = item_lhs.with_vector_size(vector_size);

    let output = scope.create_local(item);

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, output));

    output
}

pub(crate) fn index_expand_no_vec<F>(
    scope: &mut Scope,
    list: Variable,
    index: Variable,
    func: F,
) -> Variable
where
    F: Fn(IndexOperator) -> Operator,
{
    let item_lhs = list.ty;

    let class = list.pointer_class();
    let ty = item_lhs.with_vector_size(0);

    let output = scope.create_local(Type::pointer(ty, class));

    let op = func(IndexOperator {
        list,
        index,
        vector_size: 0,
        unroll_factor: 1,
    });

    scope.register(Instruction::new(op, output));

    output
}
pub(crate) fn index_expand<F, Op>(
    scope: &mut Scope,
    list: Variable,
    index: Variable,
    vector_size: Option<VectorSize>,
    func: F,
) -> Variable
where
    F: Fn(IndexOperator) -> Op,
    Op: Into<Operation>,
{
    let item_lhs = list.ty;

    let ty = if let Some(vector_size) = vector_size {
        item_lhs.with_vector_size(vector_size)
    } else {
        item_lhs
    };

    let class = list.pointer_class();
    let output = scope.create_local(Type::pointer(ty, class));

    let op = func(IndexOperator {
        list,
        index,
        vector_size: vector_size.unwrap_or(0),
        unroll_factor: 1,
    });

    scope.register(Instruction::new(op, output));

    output
}

pub(crate) fn binary_expand_fixed_output<F>(
    scope: &mut Scope,
    lhs: Variable,
    rhs: Variable,
    out_item: Type,
    func: F,
) -> Variable
where
    F: Fn(BinaryOperator) -> Arithmetic,
{
    let out = scope.create_local(out_item);
    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out));

    out
}

pub(crate) fn cmp_expand<F>(scope: &mut Scope, lhs: Variable, rhs: Variable, func: F) -> Variable
where
    F: Fn(BinaryOperator) -> Comparison,
{
    let item_lhs = lhs.ty;
    let item_rhs = rhs.ty;

    let vector_size = find_vectorization(item_lhs, item_rhs);

    let out_item = Type::scalar(ElemType::Bool).with_vector_size(vector_size);

    let out = scope.create_local(out_item);

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out));

    out
}

pub(crate) fn assign_op_expand<T: CubeType, Op>(
    scope: &mut Scope,
    lhs: &mut NativeExpand<T>,
    rhs: NativeExpand<T>,
    func: impl Fn(BinaryOperator) -> Op,
) where
    Op: Into<Operation>,
    NativeExpand<T>: ExpandDeref<Target = NativeExpand<T>>,
{
    let lhs_value = lhs.__expand_deref_method(scope).expand;
    let lhs = lhs.expand;
    let rhs = rhs.expand;

    if lhs.is_immutable() {
        panic!("Can't have a mutable operation on a const variable. Try to use `RuntimeCell`.");
    }

    let op = func(BinaryOperator {
        lhs: lhs_value,
        rhs,
    });

    scope.register(Instruction::new(op, lhs));
}

pub fn unary_expand<F, Op>(scope: &mut Scope, input: Variable, func: F) -> Variable
where
    F: Fn(UnaryOperator) -> Op,
    Op: Into<Operation>,
{
    let item = input.ty;

    let out = scope.create_local(item);

    let op = func(UnaryOperator { input });

    scope.register(Instruction::new(op, out));

    out
}

pub fn unary_expand_fixed_output<F, Op>(
    scope: &mut Scope,
    input: Variable,
    out_item: Type,
    func: F,
) -> Variable
where
    F: Fn(UnaryOperator) -> Op,
    Op: Into<Operation>,
{
    let output = scope.create_local(out_item);

    let op = func(UnaryOperator { input });

    scope.register(Instruction::new(op, output));

    output
}

pub fn init_expand<F>(scope: &mut Scope, input: Variable, mutable: bool, func: F) -> Variable
where
    F: Fn(Variable) -> Operation,
{
    let item = input.ty;

    let out = if mutable {
        scope.create_local_mut(item)
    } else {
        scope.create_local(item)
    };

    let op = func(input);
    scope.register(Instruction::new(op, out));

    out
}

pub(crate) fn find_vectorization(lhs: Type, rhs: Type) -> VectorSize {
    if matches!(lhs, Type::Scalar(_)) && matches!(rhs, Type::Scalar(_)) {
        0
    } else {
        lhs.vector_size().max(rhs.vector_size())
    }
}

pub fn assign_binary_op_expand<
    A: CubeType,
    V: CubeType,
    F: Fn(BinaryOperator) -> Op,
    Op: Into<Operation>,
>(
    scope: &mut Scope,
    lhs: &mut NativeExpand<A>,
    rhs: NativeExpand<V>,
    func: F,
) where
    NativeExpand<A>: ExpandDeref<Target = NativeExpand<A>>,
{
    let lhs_value = lhs.__expand_deref_method(scope).expand;
    let lhs: Variable = lhs.expand;
    let rhs: Variable = rhs.into();

    scope.register(Instruction::new(
        func(BinaryOperator {
            lhs: lhs_value,
            rhs,
        }),
        lhs,
    ));
}

pub trait DivCeil: Int + CubeType<ExpandType: DivCeilExpand<Self>> {
    fn div_ceil(self, divisor: Self) -> Self;

    fn __expand_div_ceil(
        scope: &mut Scope,
        a: NativeExpand<Self>,
        b: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        a.__expand_div_ceil_method(scope, b)
    }
}

pub trait DivCeilExpand<E: Int> {
    fn __expand_div_ceil_method(self, scope: &mut Scope, divisor: Self) -> Self;
}

impl<E: DivCeil> DivCeilExpand<E> for NativeExpand<E> {
    fn __expand_div_ceil_method(
        self,
        scope: &mut Scope,
        divisor: NativeExpand<E>,
    ) -> NativeExpand<E> {
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
        scope: &mut Scope,
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

use super::{CubePrimitive, Numeric, Vectorized};
use crate::{
    ir::{ConstantScalarValue, Elem, FloatKind, Item, Operator, Variable},
    prelude::{assign, init_expand, CubeContext, CubeIndex, KernelBuilder, KernelLauncher},
    Runtime,
};
use alloc::rc::Rc;
use half::{bf16, f16};
use std::{marker::PhantomData, num::NonZero};

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have ExpandElement as associated type
/// Variables whose values will be known at compile time
/// must have the primitive type as associated type
///
/// Note: Cube functions should be written using CubeTypes,
/// so that the code generated uses the associated ExpandType.
/// This allows Cube code to not necessitate cloning, which is cumbersome
/// in algorithmic code. The necessary cloning will automatically appear in
/// the generated code.
pub trait CubeType {
    type ExpandType: Clone + Init;

    /// Wrapper around the init method, necessary to type inference.
    fn init(context: &mut CubeContext, expand: Self::ExpandType) -> Self::ExpandType {
        expand.init(context)
    }
}

/// Trait useful for cube types that are also used with comptime.
pub trait IntoRuntime: CubeType + Sized {
    /// Make sure a type is actually expanded into its runtime [expand type](CubeType::ExpandType).
    fn runtime(self) -> Self {
        self
    }

    fn __expand_runtime_method(self, context: &mut CubeContext) -> Self::ExpandType;
}

/// Trait to be implemented by [cube types](CubeType) implementations.
pub trait Init: Sized {
    /// Initialize a type within a [context](CubeContext).
    ///
    /// You can return the same value when the variable is a non-mutable data structure or
    /// if the type can not be deeply cloned/copied.
    fn init(self, context: &mut CubeContext) -> Self;
}

/// Defines how a [launch argument](LaunchArg) can be expanded.
///
/// Normally this type should be implemented two times for an argument.
/// Once for the reference and the other for the mutable reference. Often time, the reference
/// should expand the argument as an input while the mutable reference should expand the argument
/// as an output.
pub trait LaunchArgExpand: CubeType {
    /// Compilation argument.
    type CompilationArg: Clone
        + PartialEq
        + Eq
        + core::hash::Hash
        + core::fmt::Debug
        + Send
        + Sync
        + 'static;

    /// Register an input variable during compilation that fill the [KernelBuilder].
    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType;
    /// Register an output variable during compilation that fill the [KernelBuilder].
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        Self::expand(arg, builder)
    }
}

/// Defines a type that can be used as argument to a kernel.
pub trait LaunchArg: LaunchArgExpand + Send + Sync + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg;
}

impl LaunchArg for () {
    type RuntimeArg<'a, R: Runtime> = ();

    fn compilation_arg<'a, R: Runtime>(
        _runtime_arg: &'a Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
    }
}

impl<R: Runtime> ArgSettings<R> for () {
    fn register(&self, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }
}

impl LaunchArgExpand for () {
    type CompilationArg = ();

    fn expand(
        _: &Self::CompilationArg,
        _builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
    }
}

impl CubeType for () {
    type ExpandType = ();
}

impl Init for () {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
}

/// Reference to a JIT variable
#[derive(Clone, Debug)]
pub enum ExpandElement {
    /// Variable kept in the variable pool.
    Managed(Rc<Variable>),
    /// Variable not kept in the variable pool.
    Plain(Variable),
}

/// Expand type associated with a type.
#[derive(new)]
pub struct ExpandElementTyped<T: CubeType> {
    pub(crate) expand: ExpandElement,
    pub(crate) _type: PhantomData<T>,
}

macro_rules! from_const {
    ($lit:ty) => {
        impl From<$lit> for ExpandElementTyped<$lit> {
            fn from(value: $lit) -> Self {
                let variable: Variable = value.into();

                ExpandElement::Plain(variable).into()
            }
        }
    };
}

from_const!(u32);
from_const!(i64);
from_const!(i32);
from_const!(f64);
from_const!(f32);
from_const!(bool);

impl From<f16> for ExpandElementTyped<f16> {
    fn from(value: f16) -> Self {
        let variable =
            Variable::ConstantScalar(ConstantScalarValue::Float(value.to_f64(), FloatKind::F16));
        ExpandElement::Plain(variable).into()
    }
}

impl From<bf16> for ExpandElementTyped<bf16> {
    fn from(value: bf16) -> Self {
        let variable =
            Variable::ConstantScalar(ConstantScalarValue::Float(value.to_f64(), FloatKind::BF16));
        ExpandElement::Plain(variable).into()
    }
}

macro_rules! tuple_cube_type {
    ($($P:ident),*) => {
        impl<$($P: CubeType),*> CubeType for ($($P,)*) {
            type ExpandType = ($($P::ExpandType,)*);
        }
    }
}
macro_rules! tuple_init {
    ($($P:ident),*) => {
        impl<$($P: Init),*> Init for ($($P,)*) {
            #[allow(non_snake_case)]
            fn init(self, context: &mut CubeContext) -> Self {
                let ($($P,)*) = self;
                ($(
                    $P.init(context),
                )*)
            }
        }
    }
}
macro_rules! tuple_runtime {
    ($($P:ident),*) => {
        impl<$($P: IntoRuntime),*> IntoRuntime for ($($P,)*) {
            #[allow(non_snake_case)]
            fn __expand_runtime_method(self, context: &mut CubeContext) -> Self::ExpandType {
                let ($($P,)*) = self;
                ($(
                    $P.__expand_runtime_method(context),
                )*)
            }
        }
    }
}

tuple_cube_type!(P1);
tuple_cube_type!(P1, P2);
tuple_cube_type!(P1, P2, P3);
tuple_cube_type!(P1, P2, P3, P4);
tuple_cube_type!(P1, P2, P3, P4, P5);
tuple_cube_type!(P1, P2, P3, P4, P5, P6);

tuple_init!(P1);
tuple_init!(P1, P2);
tuple_init!(P1, P2, P3);
tuple_init!(P1, P2, P3, P4);
tuple_init!(P1, P2, P3, P4, P5);
tuple_init!(P1, P2, P3, P4, P5, P6);

tuple_runtime!(P1);
tuple_runtime!(P1, P2);
tuple_runtime!(P1, P2, P3);
tuple_runtime!(P1, P2, P3, P4);
tuple_runtime!(P1, P2, P3, P4, P5);
tuple_runtime!(P1, P2, P3, P4, P5, P6);

pub trait ExpandElementBaseInit: CubeType {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement;
}

impl<T: ExpandElementBaseInit> Init for ExpandElementTyped<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        <T as ExpandElementBaseInit>::init_elem(context, self.into()).into()
    }
}

impl<T: CubeType> Vectorized for ExpandElementTyped<T> {
    fn vectorization_factor(&self) -> u32 {
        self.expand.vectorization_factor()
    }

    fn vectorize(self, factor: u32) -> Self {
        Self {
            expand: self.expand.vectorize(factor),
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> ExpandElementTyped<T> {
    // Expanded version of vectorization factor.
    pub fn __expand_vectorization_factor_method(self, _context: &mut CubeContext) -> u32 {
        self.expand
            .item()
            .vectorization
            .map(|it| it.get())
            .unwrap_or(1) as u32
    }

    pub fn __expand_vectorize_method(self, _context: &mut CubeContext, factor: u32) -> Self {
        Self {
            expand: self.expand.vectorize(factor),
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> Clone for ExpandElementTyped<T> {
    fn clone(&self) -> Self {
        Self {
            expand: self.expand.clone(),
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> From<ExpandElement> for ExpandElementTyped<T> {
    fn from(expand: ExpandElement) -> Self {
        Self {
            expand,
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> From<ExpandElementTyped<T>> for ExpandElement {
    fn from(value: ExpandElementTyped<T>) -> Self {
        value.expand
    }
}

impl<T: CubePrimitive> ExpandElementTyped<T> {
    /// Create an [ExpandElementTyped] from a value that is normally a literal.
    pub fn from_lit<L: Into<Variable>>(lit: L) -> Self {
        let variable: Variable = lit.into();
        let variable = T::as_elem().from_constant(variable);

        ExpandElementTyped::new(ExpandElement::Plain(variable))
    }

    /// Get the [ConstantScalarValue] from the variable.
    pub fn constant(&self) -> Option<ConstantScalarValue> {
        match *self.expand {
            Variable::ConstantScalar(val) => Some(val),
            _ => None,
        }
    }
}

impl ExpandElement {
    /// If the element can be mutated inplace, potentially reusing the register.
    pub fn can_mut(&self) -> bool {
        match self {
            ExpandElement::Managed(var) => {
                if let Variable::Local { .. } = var.as_ref() {
                    Rc::strong_count(var) <= 2
                } else {
                    false
                }
            }
            ExpandElement::Plain(_) => false,
        }
    }

    /// Explicitly consume the element, freeing it for reuse if no other copies exist.
    pub fn consume(self) -> Variable {
        *self
    }
}

impl core::ops::Deref for ExpandElement {
    type Target = Variable;

    fn deref(&self) -> &Self::Target {
        match self {
            ExpandElement::Managed(var) => var.as_ref(),
            ExpandElement::Plain(var) => var,
        }
    }
}

impl From<ExpandElement> for Variable {
    fn from(value: ExpandElement) -> Self {
        match value {
            ExpandElement::Managed(var) => *var,
            ExpandElement::Plain(var) => var,
        }
    }
}

pub(crate) fn init_expand_element<E: Into<ExpandElement>>(
    context: &mut CubeContext,
    element: E,
) -> ExpandElement {
    let elem = element.into();

    if elem.can_mut() {
        // Can reuse inplace :)
        return elem;
    }

    let mut init = |elem: ExpandElement| init_expand(context, elem, Operator::Assign);

    match *elem {
        Variable::GlobalScalar { .. } => init(elem),
        Variable::ConstantScalar { .. } => init(elem),
        Variable::Local { .. } => init(elem),
        Variable::Versioned { .. } => init(elem),
        Variable::LocalBinding { .. } => init(elem),
        // Constant should be initialized since the new variable can be mutated afterward.
        // And it is assumed those values are cloned.
        Variable::Rank
        | Variable::UnitPos
        | Variable::UnitPosX
        | Variable::UnitPosY
        | Variable::UnitPosZ
        | Variable::CubePos
        | Variable::CubePosX
        | Variable::CubePosY
        | Variable::CubePosZ
        | Variable::CubeDim
        | Variable::CubeDimX
        | Variable::CubeDimY
        | Variable::CubeDimZ
        | Variable::CubeCount
        | Variable::CubeCountX
        | Variable::CubeCountY
        | Variable::CubeCountZ
        | Variable::SubcubeDim
        | Variable::AbsolutePos
        | Variable::AbsolutePosX
        | Variable::AbsolutePosY
        | Variable::AbsolutePosZ => init(elem),
        // Array types can't be copied, so we should simply return the same variable.
        Variable::SharedMemory { .. }
        | Variable::GlobalInputArray { .. }
        | Variable::GlobalOutputArray { .. }
        | Variable::LocalArray { .. }
        | Variable::ConstantArray { .. }
        | Variable::Slice { .. }
        | Variable::Matrix { .. } => elem,
    }
}

impl Init for ExpandElement {
    fn init(self, context: &mut CubeContext) -> Self {
        init_expand_element(context, self)
    }
}

impl<T: Init> Init for Option<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        self.map(|o| Init::init(o, context))
    }
}

impl<T: CubeType> CubeType for Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: CubeType> CubeType for &mut Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: Init> Init for Vec<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        self.into_iter().map(|e| e.init(context)).collect()
    }
}

/// Create a constant element of the correct type during expansion.
pub(crate) fn __expand_new<C: Numeric, Out: Numeric>(
    _context: &mut CubeContext,
    val: C,
) -> ExpandElementTyped<Out> {
    let val = Out::from(val).unwrap();
    val.into()
}

/// Create a vectorized constant element of the correct type during expansion.
pub(crate) fn __expand_vectorized<C: Numeric + CubeIndex<u32>, Out: Numeric>(
    context: &mut CubeContext,
    val: C,
    vectorization: u32,
    elem: Elem,
) -> ExpandElementTyped<Out> {
    let new_var =
        context.create_local_binding(Item::vectorized(elem, NonZero::new(vectorization as u8)));
    let val = Out::from(val).unwrap();
    let val: ExpandElementTyped<Out> = val.into();

    // Explanation for removing all this code: Assignments are already being unrolled and broadcast
    // in the backend, so this was just duplicating code and it interfered with the SSA allocator
    assign::expand(context, val, new_var.clone().into());

    new_var.into()
}

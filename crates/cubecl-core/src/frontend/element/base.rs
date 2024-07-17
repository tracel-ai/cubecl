use std::marker::PhantomData;

use super::{Bool, Float, Int, Numeric, UInt, Vectorized, F64, I64};
use crate::{
    ir::{ConstantScalarValue, Elem, Item, Operator, Variable, Vectorization},
    prelude::{index_assign, init_expand, CubeContext, KernelBuilder, KernelLauncher},
    KernelSettings, Runtime,
};
use alloc::rc::Rc;

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

    /// Wrapper around the init method, necesary to type inference.
    fn init(context: &mut CubeContext, expand: Self::ExpandType) -> Self::ExpandType {
        expand.init(context)
    }
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
    /// Register an input variable during compilation that fill the [KernelBuilder].
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> <Self as CubeType>::ExpandType;
    /// Register an output variable during compilation that fill the [KernelBuilder].
    fn expand_output(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> <Self as CubeType>::ExpandType {
        Self::expand(builder, vectorization)
    }
}

/// Defines a type that can be used as argument to a kernel.
pub trait LaunchArg: LaunchArgExpand + Send + Sync + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;
}

impl LaunchArg for () {
    type RuntimeArg<'a, R: Runtime> = ();
}

impl<R: Runtime> ArgSettings<R> for () {
    fn register(&self, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }
}

impl LaunchArgExpand for () {
    fn expand(
        _builder: &mut KernelBuilder,
        _vectorization: Vectorization,
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
    /// Configure an input argument at the given position.
    fn configure_input(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
    /// Configure an output argument at the given position.
    fn configure_output(&self, _position: usize, settings: KernelSettings) -> KernelSettings {
        settings
    }
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

// Fake implementation for traits.
impl<T: CubeType> CubeType for ExpandElementTyped<T> {
    type ExpandType = ();
}

macro_rules! from_const {
    ($lit:ty, $ty:ty) => {
        impl From<$lit> for ExpandElementTyped<$ty> {
            fn from(value: $lit) -> Self {
                let variable: Variable = value.into();

                ExpandElement::Plain(variable).into()
            }
        }
    };
    (val $lit:ty) => {
        impl From<$lit> for ExpandElementTyped<UInt> {
            fn from(value: $lit) -> Self {
                let variable: Variable = value.val.into();

                ExpandElement::Plain(variable).into()
            }
        }
    };
}

from_const!(u32, UInt);
from_const!(i64, I64);
from_const!(f64, F64);
from_const!(bool, Bool);
from_const!(val UInt);

impl<F: Float> From<f32> for ExpandElementTyped<F> {
    fn from(value: f32) -> Self {
        ExpandElement::Plain(F::as_elem().from_constant(value.into())).into()
    }
}

impl<I: Int> From<i32> for ExpandElementTyped<I> {
    fn from(value: i32) -> Self {
        ExpandElement::Plain(I::as_elem().from_constant(value.into())).into()
    }
}

pub trait ExpandElementBaseInit: CubeType {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement;
}

impl<T: ExpandElementBaseInit> Init for ExpandElementTyped<T> {
    fn init(self, context: &mut CubeContext) -> Self {
        <T as ExpandElementBaseInit>::init_elem(context, self.into()).into()
    }
}

impl<T: CubeType> Vectorized for ExpandElementTyped<T> {
    fn vectorization_factor(&self) -> UInt {
        self.expand.vectorization_factor()
    }

    fn vectorize(self, factor: UInt) -> Self {
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

impl<T: CubeType> ExpandElementTyped<T> {
    pub fn from_lit<L: Into<Variable>>(lit: L) -> Self {
        let variable: Variable = lit.into();

        ExpandElementTyped::new(ExpandElement::Plain(variable))
    }

    pub fn constant(&self) -> Option<ConstantScalarValue> {
        match *self.expand {
            Variable::ConstantScalar(val) => Some(val),
            _ => None,
        }
    }
}

impl ExpandElement {
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
        Variable::LocalScalar { .. } => init(elem),
        Variable::ConstantScalar { .. } => init(elem),
        Variable::Local { .. } => init(elem),
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
        | Variable::Slice { .. }
        | Variable::Matrix { .. } => elem,
    }
}

impl Init for ExpandElement {
    fn init(self, context: &mut CubeContext) -> Self {
        init_expand_element(context, self)
    }
}

macro_rules! impl_init_for {
    ($($t:ty),*) => {
        $(
            impl Init for $t {
                fn init(self, _context: &mut CubeContext) -> Self {
                    panic!("Shouln't be called, only for comptime.")
                }
            }

        )*
    };
}

// Add all types used within comptime
impl_init_for!(u32, bool, UInt);

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

pub(crate) fn __expand_new<C: Numeric>(
    _context: &mut CubeContext,
    val: ExpandElementTyped<C>,
    elem: Elem,
) -> ExpandElementTyped<C> {
    ExpandElement::Plain(elem.from_constant(*val.expand)).into()
}

pub(crate) fn __expand_vectorized<C: Numeric>(
    context: &mut CubeContext,
    val: ExpandElementTyped<C>,
    vectorization: UInt,
    elem: Elem,
) -> ExpandElementTyped<C> {
    if vectorization.val == 1 {
        __expand_new(context, val, elem)
    } else {
        let new_var = context.create_local(Item::vectorized(elem, vectorization.val as u8));

        for (i, element) in vec![val; vectorization.val as usize].iter().enumerate() {
            let element = elem.from_constant(*element.expand);

            index_assign::expand::<C>(
                context,
                new_var.clone().into(),
                ExpandElementTyped::from_lit(i),
                ExpandElement::Plain(element).into(),
            );
        }

        new_var.into()
    }
}

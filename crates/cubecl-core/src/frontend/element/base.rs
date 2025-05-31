use super::{CubePrimitive, Numeric};
use crate::{
    Runtime,
    ir::{ConstantScalarValue, Operation, Scope, Variable, VariableKind},
    prelude::{KernelBuilder, KernelLauncher, init_expand},
};
use cubecl_common::{e2m1, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
use cubecl_ir::ExpandElement;
use half::{bf16, f16};
use std::{
    any::{Any, TypeId},
    marker::PhantomData,
};
use variadics_please::all_tuples;

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
#[diagnostic::on_unimplemented(note = "Consider using `#[derive(CubeType)]` on `{Self}`")]
pub trait CubeType {
    type ExpandType: Clone + IntoMut + CubeDebug;

    /// Wrapper around the init method, necessary to type inference.
    fn into_mut(scope: &mut Scope, expand: Self::ExpandType) -> Self::ExpandType {
        expand.into_mut(scope)
    }
}

/// Trait useful to convert a comptime value into runtime value.
pub trait IntoRuntime: CubeType + Sized {
    fn runtime(self) -> Self {
        self
    }

    fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType;
}

/// Convert an expand type to a version with mutable registers when necessary.
pub trait IntoMut: Sized {
    fn into_mut(self, scope: &mut Scope) -> Self;
}

pub trait CubeDebug: Sized {
    /// Set the debug name of this type's expansion. Should do nothing for types that don't appear
    /// at runtime
    #[allow(unused)]
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {}
}

/// A type that can be used as a kernel comptime argument.
/// Note that a type doesn't need to implement `CubeComptime` to be used as
/// a comptime argument. However, this facilitate the declaration of generic cube types.
///
/// # Example
///
/// ```ignore
/// #[derive(CubeType)]
/// pub struct Example<A: CubeType, B: CubeComptime> {
///     a: A,
///     #[cube(comptime)]
///     b: B
/// }
/// ```
pub trait CubeComptime: core::fmt::Debug + core::hash::Hash + Eq + Clone + Copy {}
impl<T> CubeComptime for T where T: core::fmt::Debug + core::hash::Hash + Eq + Clone + Copy {}

/// A [CubeType] that can be used as a kernel argument such as [Array] or [Tensor].
pub trait CubeLaunch: CubeType + LaunchArg + LaunchArgExpand {}
impl<T: CubeType + LaunchArg + LaunchArgExpand> CubeLaunch for T {}

/// Argument used during the compilation of kernels.
pub trait CompilationArg:
    serde::Serialize
    + serde::de::DeserializeOwned
    + Clone
    + PartialEq
    + Eq
    + core::hash::Hash
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// Compilation args should be the same even with different element types. However, it isn't
    /// possible to enforce it with the type system. So, we make the compilation args serializable
    /// and dynamically cast them.
    ///
    /// Without this, the compilation time is unreasonable. The performance drop isn't a concern
    /// since this is only done once when compiling a kernel for the first time.
    fn dynamic_cast<Arg: CompilationArg>(&self) -> Arg {
        if TypeId::of::<Arg>() == TypeId::of::<Self>() {
            let tmp: Box<dyn Any> = Box::new(self.clone());
            *tmp.downcast().unwrap()
        } else {
            let val = serde_json::to_string(self).unwrap();
            serde_json::from_str(&val)
                .expect("Compilation argument should be the same even with different element types")
        }
    }
}

impl CompilationArg for () {}

/// Defines how a [launch argument](LaunchArg) can be expanded.
///
/// TODO Verify the accuracy of the next comment.
///
/// Normally this type should be implemented two times for an argument.
/// Once for the reference and the other for the mutable reference. Often time, the reference
/// should expand the argument as an input while the mutable reference should expand the argument
/// as an output.
#[diagnostic::on_unimplemented(note = "Consider using `#[derive(CubeLaunch)]` on `{Self}`")]
pub trait LaunchArgExpand: CubeType {
    /// Compilation argument.
    type CompilationArg: CompilationArg;

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

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information of an argument to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
}

/// Expand type associated with a type.
#[derive(new)]
pub struct ExpandElementTyped<T: CubeType> {
    pub(crate) expand: ExpandElement,
    pub(crate) _type: PhantomData<T>,
}

impl<T: CubeType> From<&ExpandElementTyped<T>> for ExpandElementTyped<T> {
    fn from(value: &ExpandElementTyped<T>) -> Self {
        value.clone()
    }
}

impl<T: CubeType> From<ExpandElementTyped<T>> for Variable {
    fn from(value: ExpandElementTyped<T>) -> Self {
        value.expand.into()
    }
}

impl<T: CubeType> From<&mut ExpandElementTyped<T>> for ExpandElementTyped<T> {
    fn from(value: &mut ExpandElementTyped<T>) -> Self {
        value.clone()
    }
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

from_const!(u8);
from_const!(u16);
from_const!(u32);
from_const!(u64);
from_const!(i64);
from_const!(i8);
from_const!(i16);
from_const!(i32);
from_const!(f64);
from_const!(f16);
from_const!(bf16);
from_const!(flex32);
from_const!(tf32);
from_const!(f32);
from_const!(e2m1);
from_const!(e2m3);
from_const!(e3m2);
from_const!(e4m3);
from_const!(e5m2);
from_const!(ue8m0);
from_const!(bool);

macro_rules! tuple_cube_type {
    ($($P:ident),*) => {
        impl<$($P: CubeType),*> CubeType for ($($P,)*) {
            type ExpandType = ($($P::ExpandType,)*);
        }
    }
}
macro_rules! tuple_init {
    ($($P:ident),*) => {
        impl<$($P: IntoMut),*> IntoMut for ($($P,)*) {
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn into_mut(self, scope: &mut Scope) -> Self {
                let ($($P,)*) = self;
                ($(
                    $P.into_mut(scope),
                )*)
            }
        }
    }
}
macro_rules! tuple_debug {
    ($($P:ident),*) => {
        impl<$($P: CubeDebug),*> CubeDebug for ($($P,)*) {}
    }
}
macro_rules! tuple_runtime {
    ($($P:ident),*) => {
        impl<$($P: IntoRuntime),*> IntoRuntime for ($($P,)*) {
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType {
                let ($($P,)*) = self;
                ($(
                    $P.__expand_runtime_method(scope),
                )*)
            }
        }
    }
}

all_tuples!(tuple_cube_type, 0, 12, P);
all_tuples!(tuple_debug, 0, 12, P);
all_tuples!(tuple_init, 0, 12, P);
all_tuples!(tuple_runtime, 0, 12, P);

impl<P: CubePrimitive> CubeDebug for P {}

pub trait ExpandElementIntoMut: CubeType {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement;
}

impl<T: ExpandElementIntoMut> IntoMut for ExpandElementTyped<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        <T as ExpandElementIntoMut>::elem_into_mut(scope, self.into()).into()
    }
}

impl<T: CubeType> CubeDebug for ExpandElementTyped<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> ExpandElementTyped<T> {
    // Expanded version of vectorization factor.
    pub fn __expand_vectorization_factor_method(self, _scope: &mut Scope) -> u32 {
        self.expand
            .item
            .vectorization
            .map(|it| it.get())
            .unwrap_or(1) as u32
    }

    pub fn into_variable(self) -> Variable {
        self.expand.consume()
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
    pub fn from_lit<L: Into<Variable>>(scope: &Scope, lit: L) -> Self {
        let variable: Variable = lit.into();
        let variable = T::as_elem(scope).from_constant(variable);

        ExpandElementTyped::new(ExpandElement::Plain(variable))
    }

    /// Get the [ConstantScalarValue] from the variable.
    pub fn constant(&self) -> Option<ConstantScalarValue> {
        match self.expand.kind {
            VariableKind::ConstantScalar(val) => Some(val),
            _ => None,
        }
    }
}

pub(crate) fn into_runtime_expand_element<E: Into<ExpandElement>>(
    scope: &mut Scope,
    element: E,
) -> ExpandElement {
    let elem = element.into();

    match elem.kind {
        VariableKind::ConstantScalar { .. } => init_expand(scope, elem, false, Operation::Copy),
        _ => elem,
    }
}

pub(crate) fn into_mut_expand_element<E: Into<ExpandElement>>(
    scope: &mut Scope,
    element: E,
) -> ExpandElement {
    let elem = element.into();

    let mut init = |elem: ExpandElement| init_expand(scope, elem, true, Operation::Copy);

    match elem.kind {
        VariableKind::GlobalScalar { .. } => init(elem),
        VariableKind::ConstantScalar { .. } => init(elem),
        VariableKind::LocalMut { .. } => init(elem),
        VariableKind::Versioned { .. } => init(elem),
        VariableKind::LocalConst { .. } => init(elem),
        VariableKind::Builtin(_) => init(elem),
        VariableKind::SharedMemory { .. }
        | VariableKind::GlobalInputArray { .. }
        | VariableKind::GlobalOutputArray { .. }
        | VariableKind::LocalArray { .. }
        | VariableKind::ConstantArray { .. }
        | VariableKind::Matrix { .. }
        | VariableKind::Barrier { .. }
        | VariableKind::Pipeline { .. }
        | VariableKind::TensorMap(_) => elem,
    }
}

impl IntoMut for ExpandElement {
    fn into_mut(self, scope: &mut Scope) -> Self {
        into_mut_expand_element(scope, self)
    }
}

impl<T: IntoMut> IntoMut for Option<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        self.map(|o| IntoMut::into_mut(o, scope))
    }
}

impl<T: CubeType> CubeType for Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: CubeType> CubeType for &mut Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: IntoMut> IntoMut for Vec<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        self.into_iter().map(|e| e.into_mut(scope)).collect()
    }
}
impl<T: CubeDebug> CubeDebug for Vec<T> {}

/// Create a constant element of the correct type during expansion.
pub(crate) fn __expand_new<C: Numeric, Out: Numeric>(
    scope: &mut Scope,
    val: C,
) -> ExpandElementTyped<Out> {
    let input: ExpandElementTyped<C> = val.into();
    let const_val = input.expand.as_const().unwrap();
    let var = Variable::constant(const_val.cast_to(Out::as_elem(scope)));
    ExpandElement::Plain(var).into()
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

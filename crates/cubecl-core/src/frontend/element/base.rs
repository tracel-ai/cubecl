use super::{CubePrimitive, Numeric};
use crate::{
    ir::{ConstantScalarValue, Operation, Scope, Variable, VariableKind},
    prelude::{KernelBuilder, KernelLauncher, init_expand},
};
use cubecl_common::{e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
use cubecl_ir::ExpandElement;
use cubecl_runtime::runtime::Runtime;
use half::{bf16, f16};
use std::marker::PhantomData;
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

/// Argument used during the compilation of kernels.
pub trait CompilationArg:
    Clone + PartialEq + Eq + core::hash::Hash + core::fmt::Debug + Send + Sync + 'static
{
    /// Compilation args should be the same even with different element types. However, it isn't
    /// possible to enforce it with the type system. So, we make the compilation args serializable
    /// and dynamically cast them.
    ///
    /// Without this, the compilation time is unreasonable. The performance drop isn't a concern
    /// since this is only done once when compiling a kernel for the first time.
    fn dynamic_cast<Arg: CompilationArg>(&self) -> Arg {
        // Dynamic cast, unlike transmute it does not require statically proving the types are the
        // same size. We assert at runtime to avoid undefined behaviour and help the compiler optimize.
        assert!(size_of::<Arg>() == size_of::<Self>());
        let this = Box::new(self.clone());
        unsafe { *Box::from_raw(Box::into_raw(this) as *mut Arg) }
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
pub trait LaunchArg: CubeType + Send + Sync + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<'a, R: Runtime>: ArgSettings<R>;
    /// Compilation argument.
    type CompilationArg: CompilationArg;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg;

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

/// Defines the argument settings used to launch a kernel.
pub trait ArgSettings<R: Runtime>: Send + Sync {
    /// Register the information of an argument to the [KernelLauncher].
    fn register(&self, launcher: &mut KernelLauncher<R>);
}

macro_rules! launch_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: LaunchArg),*> LaunchArg for ($($T),*) {
            type RuntimeArg<'a, R: Runtime> = ($($T::RuntimeArg<'a, R>),*);
            type CompilationArg = ($($T::CompilationArg),*);

            fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
                let ($($t),*) = runtime_arg;
                ($($T::compilation_arg($t)),*)
            }

            fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> ($(<$T as CubeType>::ExpandType),*) {
                let ($($t),*) = arg;
                ($($T::expand($t, builder)),*)
            }

            fn expand_output(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> ($(<$T as CubeType>::ExpandType),*) {
                let ($($t),*) = arg;
                ($($T::expand_output($t, builder)),*)
            }
        }

        impl<$($T: CompilationArg),*> CompilationArg for ($($T),*) {}

        impl<R: Runtime, $($T: ArgSettings<R>),*> ArgSettings<R> for ($($T),*) {
            fn register(&self, launcher: &mut KernelLauncher<R>) {
                let ($($t),*) = self;
                $($t.register(launcher);)*
            }
        }
    };
}

all_tuples!(launch_tuple, 2, 12, T, t);

/// Expand type associated with a type.
#[derive(new)]
pub struct ExpandElementTyped<T: CubeType> {
    pub expand: ExpandElement,
    pub(crate) _type: PhantomData<T>,
}

impl<T: CubeType> ExpandElementTyped<T> {
    /// Casts a reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `ExpandElement`
    pub unsafe fn as_type_ref_unchecked<E: CubeType>(&self) -> &ExpandElementTyped<E> {
        unsafe { core::mem::transmute::<&ExpandElementTyped<T>, &ExpandElementTyped<E>>(self) }
    }

    /// Casts a mutable reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `ExpandElement`
    pub unsafe fn as_type_mut_unchecked<E: CubeType>(&mut self) -> &mut ExpandElementTyped<E> {
        unsafe {
            core::mem::transmute::<&mut ExpandElementTyped<T>, &mut ExpandElementTyped<E>>(self)
        }
    }
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
from_const!(e2m1x2);
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

impl<T: CubeType> CubeDebug for &ExpandElementTyped<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> CubeDebug for &mut ExpandElementTyped<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> ExpandElementTyped<T> {
    /// Comptime version of [size](Array::line_size).
    pub fn line_size(&self) -> u32 {
        self.expand.ty.line_size()
    }

    // Expanded version of vectorization factor.
    pub fn __expand_line_size_method(self, _scope: &mut Scope) -> u32 {
        self.expand.ty.line_size()
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
        let variable = T::as_type(scope).from_constant(variable);

        ExpandElementTyped::new(ExpandElement::Plain(variable))
    }

    /// Get the [ConstantScalarValue] from the variable.
    pub fn constant(&self) -> Option<ConstantScalarValue> {
        match self.expand.kind {
            VariableKind::ConstantScalar(val) => Some(val),
            _ => None,
        }
    }

    pub fn __expand_into_lit_unchecked_method(self, _scope: &mut Scope) -> T {
        let value = self.constant().unwrap();
        T::from_const_value(value)
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
        VariableKind::Shared { .. }
        | VariableKind::SharedArray { .. }
        | VariableKind::GlobalInputArray { .. }
        | VariableKind::GlobalOutputArray { .. }
        | VariableKind::LocalArray { .. }
        | VariableKind::ConstantArray { .. }
        | VariableKind::Matrix { .. }
        | VariableKind::BarrierToken { .. }
        | VariableKind::Pipeline { .. }
        | VariableKind::TensorMapOutput(_)
        | VariableKind::TensorMapInput(_) => elem,
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
    let var = Variable::constant(const_val.cast_to(Out::as_type(scope)));
    ExpandElement::Plain(var).into()
}

impl LaunchArg for () {
    type RuntimeArg<'a, R: Runtime> = ();
    type CompilationArg = ();

    fn compilation_arg<'a, R: Runtime>(
        _runtime_arg: &'a Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
    }

    fn expand(
        _: &Self::CompilationArg,
        _builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
    }
}

impl<R: Runtime> ArgSettings<R> for () {
    fn register(&self, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }
}

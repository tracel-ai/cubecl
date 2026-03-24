use super::{CubePrimitive, Numeric};
use crate::{
    ir::{ConstantValue, Scope, Variable, VariableKind},
    prelude::{DynamicSize, KernelBuilder, KernelLauncher, assign},
    unexpanded,
};
use alloc::{boxed::Box, vec::Vec};
use core::marker::PhantomData;
use cubecl_common::{e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
use cubecl_ir::{ManagedVariable, VectorSize};
use cubecl_runtime::runtime::Runtime;
use half::{bf16, f16};
use variadics_please::{all_tuples, all_tuples_enumerated};

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have `ManagedVariable` as associated type
/// Variables whose values will be known at compile time
/// must have the primitive type as associated type
///
/// Note: Cube functions should be written using `CubeTypes`,
/// so that the code generated uses the associated `ExpandType`.
/// This allows Cube code to not necessitate cloning, which is cumbersome
/// in algorithmic code. The necessary cloning will automatically appear in
/// the generated code.
#[diagnostic::on_unimplemented(note = "Consider using `#[derive(CubeType)]` on `{Self}`")]
pub trait CubeType {
    type ExpandType: Clone + IntoMut + CubeDebug;
}

pub trait CubeEnum: Sized {
    type RuntimeValue: Clone + CubeDebug;

    fn discriminant(&self) -> NativeExpand<i32>;

    /// Return the runtime value of this enum, if only one variant has a value.
    /// Should return () for all other cases.
    fn runtime_value(self) -> Self::RuntimeValue;

    fn discriminant_of_value(&self, variant_name: &'static str) -> i32 {
        Self::discriminant_of(variant_name)
    }

    fn discriminant_of(variant_name: &'static str) -> i32;
}

pub trait Assign {
    /// Assign `value` to `self` in `scope`.
    fn expand_assign(&mut self, scope: &mut Scope, value: Self);
    /// Create a new mutable variable of this type in `scope`.
    fn init_mut(&self, scope: &mut Scope) -> Self;
}

impl<T: CubePrimitive> Assign for T {
    fn expand_assign(&mut self, _scope: &mut Scope, value: Self) {
        *self = value;
    }
    fn init_mut(&self, _scope: &mut Scope) -> Self {
        *self
    }
}

impl<T: NativeAssign> Assign for NativeExpand<T> {
    fn expand_assign(&mut self, scope: &mut Scope, value: Self) {
        assign::expand(scope, value, self.clone());
    }
    fn init_mut(&self, scope: &mut Scope) -> Self {
        T::elem_init_mut(scope, self.expand.clone()).into()
    }
}

impl<T: Assign> Assign for Option<T> {
    fn expand_assign(&mut self, scope: &mut Scope, value: Self) {
        match (self, value) {
            (Some(this), Some(other)) => this.expand_assign(scope, other),
            (None, None) => {}
            _ => panic!("Can't assign mismatched enum variants"),
        }
    }
    fn init_mut(&self, scope: &mut Scope) -> Self {
        self.as_ref().map(|value| value.init_mut(scope))
    }
}

impl<T: Assign> Assign for Vec<T> {
    fn expand_assign(&mut self, scope: &mut Scope, value: Self) {
        assert!(
            self.len() == value.len(),
            "Can't assign mismatched vector lengths"
        );
        for (this, other) in self.iter_mut().zip(value) {
            this.expand_assign(scope, other);
        }
    }
    fn init_mut(&self, scope: &mut Scope) -> Self {
        self.iter().map(|it| it.init_mut(scope)).collect()
    }
}

pub trait CloneExpand {
    fn __expand_clone_method(&self, scope: &mut Scope) -> Self;
}

impl<C: Clone> CloneExpand for C {
    fn __expand_clone_method(&self, _scope: &mut Scope) -> Self {
        self.clone()
    }
}

/// Trait useful to convert a comptime value into runtime value.
pub trait IntoRuntime: CubeType + Sized {
    fn runtime(self) -> Self {
        self
    }

    fn __expand_runtime_method(self, scope: &mut Scope) -> Self::ExpandType;
}

/// Trait for marking a function return value as comptime when the compiler can't infer it.
pub trait IntoComptime: Sized {
    #[allow(clippy::wrong_self_convention)]
    fn comptime(self) -> Self {
        self
    }
}

impl<T: Sized> IntoComptime for T {}

/// Convert an expand type to a version with mutable registers when necessary.
pub trait IntoMut: Sized {
    /// Convert the variable into a potentially new mutable variable in `scope`, copying if needed.
    fn into_mut(self, scope: &mut Scope) -> Self;
}

pub fn into_mut_assign<T: Assign>(value: T, scope: &mut Scope) -> T {
    let mut out = value.init_mut(scope);
    out.expand_assign(scope, value);
    out
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

impl<T: Clone + PartialEq + Eq + core::hash::Hash + core::fmt::Debug + Send + Sync + 'static>
    CompilationArg for T
{
}

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
    type RuntimeArg<R: Runtime>: Send + Sync;
    /// Compilation argument.
    type CompilationArg: CompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg;

    /// Register an input variable during compilation that fill the [`KernelBuilder`].
    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType;

    /// Register an output variable during compilation that fill the [`KernelBuilder`].
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        Self::expand(arg, builder)
    }
}

macro_rules! launch_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: LaunchArg),*> LaunchArg for ($($T),*) {
            type RuntimeArg<R: Runtime> = ($($T::RuntimeArg<R>),*);
            type CompilationArg = ($($T::CompilationArg),*);

            fn register<R: Runtime>(runtime_arg: Self::RuntimeArg<R>, launcher: &mut KernelLauncher<R>) -> Self::CompilationArg {
                let ($($t),*) = runtime_arg;
                ($($T::register($t, launcher)),*)
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
    };
}

all_tuples!(launch_tuple, 2, 12, T, t);

/// Expand type of a native GPU type, i.e. scalar primitives, arrays, shared memory.
#[derive(new)]
pub struct NativeExpand<T: CubeType> {
    pub expand: ManagedVariable,
    pub(crate) _type: PhantomData<T>,
}

impl<T: CubeType> NativeExpand<T> {
    /// Casts a reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `ManagedVariable`
    pub unsafe fn as_type_ref_unchecked<E: CubeType>(&self) -> &NativeExpand<E> {
        unsafe { core::mem::transmute::<&NativeExpand<T>, &NativeExpand<E>>(self) }
    }

    /// Casts a mutable reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `ManagedVariable`
    pub unsafe fn as_type_mut_unchecked<E: CubeType>(&mut self) -> &mut NativeExpand<E> {
        unsafe { core::mem::transmute::<&mut NativeExpand<T>, &mut NativeExpand<E>>(self) }
    }
}

impl<T: CubeType> From<&NativeExpand<T>> for NativeExpand<T> {
    fn from(value: &NativeExpand<T>) -> Self {
        value.clone()
    }
}

impl<T: CubeType> From<NativeExpand<T>> for Variable {
    fn from(value: NativeExpand<T>) -> Self {
        value.expand.into()
    }
}

impl<T: CubeType> From<&mut NativeExpand<T>> for NativeExpand<T> {
    fn from(value: &mut NativeExpand<T>) -> Self {
        value.clone()
    }
}

macro_rules! from_const {
    ($lit:ty) => {
        impl From<$lit> for NativeExpand<$lit> {
            fn from(value: $lit) -> Self {
                let variable: Variable = value.into();

                ManagedVariable::Plain(variable).into()
            }
        }
    };
}

from_const!(u8);
from_const!(u16);
from_const!(u32);
from_const!(u64);
from_const!(usize);
from_const!(isize);
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
macro_rules! tuple_assign {
    ($(($n: tt, $P:ident)),*) => {
        impl<$($P: Assign),*> Assign for ($($P,)*) {
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn expand_assign(&mut self, scope: &mut Scope, value: Self) {
                let ($($P,)*) = self;
                $(
                    $P.expand_assign(scope, value.$n);
                )*
            }
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn init_mut(&self, scope: &mut Scope) -> Self {
                let ($($P,)*) = self;
                ($(
                    $P.init_mut(scope),
                )*)
            }
        }
    }
}

all_tuples!(tuple_cube_type, 0, 12, P);
all_tuples!(tuple_debug, 0, 12, P);
all_tuples!(tuple_init, 0, 12, P);
all_tuples!(tuple_runtime, 0, 12, P);
all_tuples_enumerated!(tuple_assign, 0, 12, P);

impl<P: CubePrimitive> CubeDebug for P {}

/// Trait for native types that can be assigned. For non-native composites, use the normal [`Assign`].
pub trait NativeAssign: CubeType {
    fn elem_init_mut(scope: &mut Scope, elem: ManagedVariable) -> ManagedVariable {
        init_mut_expand_element(scope, &elem)
    }
}

impl<T: NativeAssign> IntoMut for NativeExpand<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        into_mut_assign(self, scope)
    }
}

impl<T: CubeType> CubeDebug for NativeExpand<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> CubeDebug for &NativeExpand<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> CubeDebug for &mut NativeExpand<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.expand, name);
    }
}

impl<T: CubeType> NativeExpand<T> {
    /// Comptime version of [`crate::frontend::Array::vector_size`].
    pub fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }

    // Expanded version of vectorization factor.
    pub fn __expand_vector_size_method(self, _scope: &mut Scope) -> VectorSize {
        self.expand.ty.vector_size()
    }

    pub fn into_variable(self) -> Variable {
        self.expand.consume()
    }
}

impl<T: CubeType> Clone for NativeExpand<T> {
    fn clone(&self) -> Self {
        Self {
            expand: self.expand.clone(),
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> From<ManagedVariable> for NativeExpand<T> {
    fn from(expand: ManagedVariable) -> Self {
        Self {
            expand,
            _type: PhantomData,
        }
    }
}

impl<T: CubeType> From<NativeExpand<T>> for ManagedVariable {
    fn from(value: NativeExpand<T>) -> Self {
        value.expand
    }
}

impl<T: CubePrimitive> NativeExpand<T> {
    /// Create an [`NativeExpand`] from a value that is normally a literal.
    pub fn from_lit<L: Into<ConstantValue>>(scope: &Scope, lit: L) -> Self {
        let variable: ConstantValue = lit.into();
        let variable = T::as_type(scope).constant(variable);

        NativeExpand::new(ManagedVariable::Plain(variable))
    }

    /// Get the [`ConstantValue`] from the variable.
    pub fn constant(&self) -> Option<ConstantValue> {
        match self.expand.kind {
            VariableKind::Constant(val) => Some(val),
            _ => None,
        }
    }

    pub fn __expand_into_lit_unchecked_method(self, _scope: &mut Scope) -> T {
        let value = self.constant().unwrap();
        T::from_const_value(value)
    }
}

pub(crate) fn init_mut_expand_element(
    scope: &mut Scope,
    element: &ManagedVariable,
) -> ManagedVariable {
    scope.create_local_mut(element.ty)
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
) -> NativeExpand<Out> {
    let input: ConstantValue = val.into();
    let var = Out::as_type(scope).constant(input);
    ManagedVariable::Plain(var).into()
}

impl LaunchArg for () {
    type RuntimeArg<R: Runtime> = ();
    type CompilationArg = ();

    fn register<R: Runtime>(_runtime_arg: Self::RuntimeArg<R>, _launcher: &mut KernelLauncher<R>) {
        // nothing to do
    }

    fn expand(
        _: &Self::CompilationArg,
        _builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
    }
}

pub trait DefaultExpand: CubeType {
    fn __expand_default(scope: &mut Scope) -> Self::ExpandType;
}

impl<T: CubeType + Default + IntoRuntime> DefaultExpand for T {
    fn __expand_default(scope: &mut Scope) -> T::ExpandType {
        T::default().__expand_runtime_method(scope)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Const<const N: usize>;

pub trait Size: core::fmt::Debug + Clone + Copy + Send + Sync + 'static {
    fn __expand_value(scope: &Scope) -> usize;
    fn value() -> usize {
        unexpanded!()
    }
    fn try_value_const() -> Option<usize> {
        None
    }
}

impl<const VALUE: usize> Size for Const<VALUE> {
    fn __expand_value(_scope: &Scope) -> usize {
        VALUE
    }
    fn value() -> usize {
        VALUE
    }
    fn try_value_const() -> Option<usize> {
        Some(VALUE)
    }
}

impl<Marker: 'static> Size for DynamicSize<Marker> {
    fn __expand_value(scope: &Scope) -> usize {
        scope.resolve_size::<Self>().expect("Size to be registered")
    }
    fn value() -> usize {
        unexpanded!()
    }
}

/// Define a custom type to be used for a comptime scalar type.
/// Useful for cases where generics can't work.
#[macro_export]
macro_rules! define_scalar {
    ($vis: vis $name: ident) => {
        $crate::__private::paste! {
            $vis struct [<__ $name>];
            $vis type $name = $crate::prelude::DynamicScalar<[<__ $name>]>;
        }
    };
}

/// Define a custom type to be used for a comptime size. Useful for cases where generics can't work.
#[macro_export]
macro_rules! define_size {
    ($vis: vis $name: ident) => {
        $crate::__private::paste! {
            $vis struct [<__ $name>];
            $vis type $name = $crate::prelude::DynamicSize<[<__ $name>]>;
        }
    };
}

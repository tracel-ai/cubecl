use super::{CubePrimitive, Numeric};
use crate::{
    frontend::read_variable,
    ir::{ConstantValue, Scope, Variable, VariableKind},
    prelude::{DynamicSize, KernelBuilder, KernelLauncher, Scalar, assign},
    unexpanded,
};
use alloc::{boxed::Box, vec::Vec};
use core::{fmt::Debug, marker::PhantomData};
use cubecl_common::{e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
use cubecl_ir::{AddressSpace, Instruction, Memory, Type, VectorSize};
use cubecl_runtime::runtime::Runtime;
use half::{bf16, f16};
use variadics_please::{all_tuples, all_tuples_enumerated};

/// Types used in a cube function must implement this trait
///
/// Variables whose values will be known at runtime must
/// have `Variable` as associated type
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
    type ExpandType: IntoExpand<Expand = Self::ExpandType>
        + ExpandTypeClone
        + IntoMut
        + CubeDebug
        + AsRefExpand
        + AsMutExpand;
}

pub trait NativeCubeType: CubeType<ExpandType = NativeExpand<Self>> {}

impl<'a, T: CubeType + ?Sized> CubeType for &'a T {
    type ExpandType = &'a T::ExpandType;
}

impl<'a, T: CubeType + ?Sized> CubeType for &'a mut T {
    type ExpandType = &'a mut T::ExpandType;
}

impl<T: CubeType + ?Sized> CubeType for *const T {
    type ExpandType = *const T::ExpandType;
}

impl<T: CubeType + ?Sized> CubeType for *mut T {
    type ExpandType = *mut T::ExpandType;
}

impl<T: CubeType<ExpandType = NativeExpand<T>> + ?Sized> NativeCubeType for T {}

pub trait IntoExpand {
    type Expand;
    fn into_expand(self, scope: &Scope) -> Self::Expand;
}

impl<'a, T: IntoExpand<Expand = T> + ?Sized> IntoExpand for &'a T {
    type Expand = &'a T;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<'a, T: IntoExpand<Expand = T> + ?Sized> IntoExpand for &'a mut T {
    type Expand = &'a mut T;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<T: IntoExpand<Expand = T> + ?Sized> IntoExpand for *const T {
    type Expand = *const T;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<T: IntoExpand<Expand = T> + ?Sized> IntoExpand for *mut T {
    type Expand = *mut T;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

pub trait ExpandTypeClone {
    /// Unchecked clone that only clones the conceptual runtime value. Should only be used in cases
    /// where each copy is used in a mutually exclusive branch (i.e. match, runtime enums). This is
    /// intentionally separated from Rust's `Clone` semantics and should only be used for the
    /// conceptual expand values, never real data. Using two values in the same branch is undefined
    /// behaviour.
    fn clone_unchecked(&self) -> Self;
}

impl<T: ExpandTypeClone + ?Sized> ExpandTypeClone for &T {
    fn clone_unchecked(&self) -> Self {
        self
    }
}

impl<T: ExpandTypeClone + ?Sized> ExpandTypeClone for &mut T {
    #[allow(mutable_transmutes)]
    fn clone_unchecked(&self) -> Self {
        unsafe { core::mem::transmute(&**self) }
    }
}

impl<T: ExpandTypeClone + ?Sized> ExpandTypeClone for *const T {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl<T: ExpandTypeClone + ?Sized> ExpandTypeClone for *mut T {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

/// Expand version of [`AsRef`](core::convert::AsRef). Like [`AsRef<Self>`](core::convert::AsRef)
/// it's implemented for all [`ExpandType`](CubeType::ExpandType)s. This is called when the Rust
/// code uses `&x`.
pub trait AsRefExpand<T: ?Sized = Self> {
    fn __expand_as_ref_method(&self, scope: &Scope) -> &T {
        self.__expand_ref_method(scope)
    }
    fn __expand_ref_method(&self, scope: &Scope) -> &T;
}

impl<T: AsRefExpand + ?Sized> AsRefExpand for &T {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl<T: AsRefExpand + ?Sized> AsRefExpand for &mut T {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl<T: AsRefExpand + ?Sized> AsRefExpand for *const T {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl<T: AsRefExpand + ?Sized> AsRefExpand for *mut T {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

/// Expand version of [`AsMut`](core::convert::AsMut). The `Self` version must be implemented by
/// all [`ExpandType`](CubeType::ExpandType)s, since `CubeCL` also uses it to implement `&mut x`.
pub trait AsMutExpand<T: ?Sized = Self> {
    fn __expand_as_mut_method(&mut self, scope: &Scope) -> &mut T {
        self.__expand_ref_mut_method(scope)
    }
    fn __expand_ref_mut_method(&mut self, scope: &Scope) -> &mut T;
}

impl<T: AsMutExpand + ?Sized> AsMutExpand for &T {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<T: AsMutExpand + ?Sized> AsMutExpand for &mut T {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<T: AsMutExpand + ?Sized> AsMutExpand for *const T {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<T: AsMutExpand + ?Sized> AsMutExpand for *mut T {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

/// `CubeCL` version of [`Deref`](core::ops::Deref). Unlike those traits, this trait produces owned
/// values directly. Maps to `*x`.
pub trait DerefExpand {
    type Target;

    fn __expand_deref_method(&self, scope: &Scope) -> Self::Target;
}

pub trait AsDerefExpand {
    type Target;
    fn __expand_as_deref_method(&self, scope: &Scope) -> &Self::Target;
}

pub trait AsDerefMutExpand: AsDerefExpand {
    fn __expand_as_deref_mut_method(&mut self, scope: &Scope) -> &mut Self::Target;
}

impl<T> AsDerefExpand for &mut T {
    type Target = T;
    fn __expand_as_deref_method(&self, _: &Scope) -> &T {
        self
    }
}

pub trait CubeEnum: Sized {
    type RuntimeValue: ExpandTypeClone + CubeDebug;

    fn discriminant(&self) -> NativeExpand<i32>;

    /// Return the runtime value of this enum, if only one variant has a value.
    /// Should return () for all other cases.
    fn runtime_value(self) -> Self::RuntimeValue;

    fn discriminant_of_value(&self, variant_name: &'static str) -> i32 {
        Self::discriminant_of(variant_name)
    }

    fn discriminant_of(variant_name: &'static str) -> i32;
}

pub trait Assign<T = Self> {
    /// Assign `value` to `self` in `scope`.
    fn __expand_assign_method(&mut self, scope: &Scope, value: T);
    /// Create a new mutable variable of this type in `scope`.
    fn init_mut(&self, scope: &Scope) -> Self;
}

impl<T: CubePrimitive> Assign for T {
    fn __expand_assign_method(&mut self, _scope: &Scope, value: Self) {
        *self = value;
    }
    fn init_mut(&self, _scope: &Scope) -> Self {
        *self
    }
}

impl<T: NativeAssign> Assign for NativeExpand<T> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        assign::expand(scope, value, self);
    }
    fn init_mut(&self, scope: &Scope) -> Self {
        T::elem_init_mut(scope, self.expand).into()
    }
}

impl<T: Assign> Assign for Option<T> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        match (self, value) {
            (Some(this), Some(other)) => this.__expand_assign_method(scope, other),
            (None, None) => {}
            _ => panic!("Can't assign mismatched enum variants"),
        }
    }
    fn init_mut(&self, scope: &Scope) -> Self {
        self.as_ref().map(|value| value.init_mut(scope))
    }
}

impl<T: Assign> Assign for Vec<T> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        assert!(
            self.len() == value.len(),
            "Can't assign mismatched vector lengths"
        );
        for (this, other) in self.iter_mut().zip(value) {
            this.__expand_assign_method(scope, other);
        }
    }
    fn init_mut(&self, scope: &Scope) -> Self {
        self.iter().map(|it| it.init_mut(scope)).collect()
    }
}

pub trait CloneExpand {
    fn __expand_clone_method(&self, scope: &Scope) -> Self;
}
impl<T: Clone> CloneExpand for T {
    fn __expand_clone_method(&self, _: &Scope) -> Self {
        self.clone()
    }
}

/// Trait useful to convert a comptime value into runtime value.
pub trait IntoRuntime:
    IntoExpand<Expand = <Self as CubeType>::ExpandType> + CubeType + Sized
{
    fn runtime(self) -> Self {
        self
    }

    fn __expand_runtime_method(self, scope: &Scope) -> Self::ExpandType;
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
    fn into_mut(self, scope: &Scope) -> Self;
}

impl<T: IntoMut> IntoMut for &T {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl<T: IntoMut> IntoMut for &mut T {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl<T: IntoMut> IntoMut for *const T {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl<T: IntoMut> IntoMut for *mut T {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

pub fn into_mut_assign<T: Assign>(value: T, scope: &Scope) -> T {
    let mut out = value.init_mut(scope);
    out.__expand_assign_method(scope, value);
    out
}

pub trait CubeDebug {
    /// Set the debug name of this type's expansion. Should do nothing for types that don't appear
    /// at runtime
    #[allow(unused)]
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {}
}

impl<T: CubeDebug + ?Sized> CubeDebug for &T {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        T::set_debug_name(self, scope, name);
    }
}

impl<T: CubeDebug + ?Sized> CubeDebug for &mut T {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        T::set_debug_name(self, scope, name);
    }
}

impl<T: CubeDebug + ?Sized> CubeDebug for *const T {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        T::set_debug_name(unsafe { &**self }, scope, name);
    }
}

impl<T: CubeDebug + ?Sized> CubeDebug for *mut T {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        T::set_debug_name(unsafe { &**self }, scope, name);
    }
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
pub trait LaunchArg: CubeType + 'static {
    /// The runtime argument for the kernel.
    type RuntimeArg<R: Runtime>: Send + Sync;
    /// Compilation argument.
    type CompilationArg: CompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg;

    /// Register a variable during compilation that fill the [`KernelBuilder`].
    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_launch_arg_ref {
    ($ty: ty) => {
        impl<T: LaunchArg + ?Sized + 'static> LaunchArg for $ty {
            type RuntimeArg<R: Runtime> = T::RuntimeArg<R>;
            type CompilationArg = T::CompilationArg;

            fn register<R: Runtime>(
                arg: Self::RuntimeArg<R>,
                launcher: &mut KernelLauncher<R>,
            ) -> Self::CompilationArg {
                T::register(arg, launcher)
            }

            fn expand(
                arg: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> <Self as CubeType>::ExpandType {
                let value = T::expand(arg, builder);
                builder.scope.create_kernel_ref(value)
            }
        }
    };
}

impl_launch_arg_ref!(&'static T);
impl_launch_arg_ref!(&'static mut T);
impl_launch_arg_ref!(*const T);
impl_launch_arg_ref!(*mut T);

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
        }
    };
}

all_tuples!(launch_tuple, 2, 12, T, t);

macro_rules! as_ref_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: AsRefExpand),*> AsRefExpand for ($($T),*) {
            fn __expand_ref_method(&self, _: &Scope) -> &($($T),*) {
                self
            }
        }
    };
}

all_tuples!(as_ref_tuple, 2, 12, T, t);

macro_rules! as_mut_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: AsMutExpand),*> AsMutExpand for ($($T),*) {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut ($($T),*) {
                self
            }
        }
    };
}

all_tuples!(as_mut_tuple, 2, 12, T, t);

macro_rules! deref_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: DerefExpand),*> DerefExpand for ($($T),*) {
            type Target = ($($T::Target),*);

            fn __expand_deref_method(&self, scope: &Scope) -> Self::Target {
                let ($($t),*) = self;
                ($($t.__expand_deref_method(scope)),*)
            }
        }
    };
}

all_tuples!(deref_tuple, 2, 12, T, t);

/// Expand type of a native GPU type, i.e. scalar primitives, arrays, shared memory.
#[derive(new, Clone, Copy, Debug)]
pub struct NativeExpand<T: ?Sized> {
    pub expand: Variable,
    pub(crate) _type: PhantomData<T>,
}

impl<T: ?Sized> IntoExpand for NativeExpand<T> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<T: ?Sized> ExpandTypeClone for NativeExpand<T> {
    fn clone_unchecked(&self) -> Self {
        NativeExpand {
            expand: self.expand,
            _type: PhantomData,
        }
    }
}

impl<T: ?Sized> NativeExpand<T> {
    /// Casts a reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `Variable`
    pub unsafe fn as_type_ref_unchecked<E: ?Sized>(&self) -> &NativeExpand<E> {
        unsafe { core::mem::transmute::<&NativeExpand<T>, &NativeExpand<E>>(self) }
    }

    /// Casts a mutable reference of this expand element to a different type.
    /// # Safety
    /// There's no guarantee the new type is valid for the `Variable`
    pub unsafe fn as_type_mut_unchecked<E: ?Sized>(&mut self) -> &mut NativeExpand<E> {
        unsafe { core::mem::transmute::<&mut NativeExpand<T>, &mut NativeExpand<E>>(self) }
    }
}

impl<T: ?Sized> AsRefExpand for NativeExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

#[diagnostic::do_not_recommend]
impl<T: CubePrimitive> AsMutExpand for NativeExpand<T> {
    fn __expand_ref_mut_method(&mut self, scope: &Scope) -> &mut Self {
        assert!(
            self.expand.can_mutate(),
            "Can't create mutable reference to immutable variable"
        );
        let ptr = scope.create_local(Type::pointer(self.expand.ty, AddressSpace::Local));
        scope.register(Instruction::new(Memory::Reference(self.expand), ptr));
        scope.create_kernel_ref(ptr.into())
    }
}

impl<T: CubePrimitive> DerefExpand for NativeExpand<T> {
    type Target = Self;

    fn __expand_deref_method(&self, scope: &Scope) -> NativeExpand<T> {
        read_variable(scope, self.expand).into()
    }
}

impl<T: ?Sized> From<NativeExpand<T>> for Variable {
    fn from(value: NativeExpand<T>) -> Self {
        value.expand
    }
}

macro_rules! from_const {
    ($lit:ty) => {
        impl From<$lit> for NativeExpand<$lit> {
            fn from(value: $lit) -> Self {
                let variable: Variable = value.into();

                variable.into()
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

        impl<$($P: IntoExpand),*> IntoExpand for ($($P,)*) {
            type Expand = ($($P::Expand,)*);

            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn into_expand(self, scope: &Scope) -> Self::Expand {
                let ($($P,)*) = self;
                ($(
                    $P.into_expand(scope),
                )*)
            }
        }

        impl<$($P: ExpandTypeClone),*> ExpandTypeClone for ($($P,)*) {
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn clone_unchecked(&self) -> Self {
                let ($($P,)*) = self;
                ($(
                    $P.clone_unchecked(),
                )*)
            }
        }
    }
}
macro_rules! tuple_init {
    ($($P:ident),*) => {
        impl<$($P: IntoMut),*> IntoMut for ($($P,)*) {
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn into_mut(self, scope: &Scope) -> Self {
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
            fn __expand_runtime_method(self, scope: &Scope) -> Self::ExpandType {
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
            fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
                let ($($P,)*) = self;
                $(
                    $P.__expand_assign_method(scope, value.$n);
                )*
            }
            #[allow(non_snake_case, unused, clippy::unused_unit)]
            fn init_mut(&self, scope: &Scope) -> Self {
                let ($($P,)*) = self;
                ($(
                    $P.init_mut(scope),
                )*)
            }
        }
    }
}

all_tuples!(tuple_cube_type, 2, 12, P);
all_tuples!(tuple_debug, 2, 12, P);
all_tuples!(tuple_init, 2, 12, P);
all_tuples!(tuple_runtime, 2, 12, P);
all_tuples_enumerated!(tuple_assign, 2, 12, P);

/// Trait for native types that can be assigned. For non-native composites, use the normal [`Assign`].
pub trait NativeAssign: CubeType {
    fn elem_init_mut(scope: &Scope, elem: Variable) -> Variable {
        init_mut_expand_element(scope, &elem)
    }
}

impl<T: NativeAssign> IntoMut for NativeExpand<T> {
    fn into_mut(self, scope: &Scope) -> Self {
        into_mut_assign(self, scope)
    }
}

impl<T: ?Sized> CubeDebug for NativeExpand<T> {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        scope.update_variable_name(self.expand, name);
    }
}

impl<T> NativeExpand<T> {
    /// Comptime version of [`crate::frontend::Array::vector_size`].
    pub fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }

    // Expanded version of vectorization factor.
    pub fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.expand.ty.vector_size()
    }

    pub fn into_variable(self) -> Variable {
        self.expand
    }
}

impl<T: ?Sized> From<Variable> for NativeExpand<T> {
    fn from(expand: Variable) -> Self {
        Self {
            expand,
            _type: PhantomData,
        }
    }
}

impl<T: Scalar + Into<ConstantValue>> NativeExpand<T> {
    /// Create an [`NativeExpand`] from a value that is normally a literal.
    pub fn from_lit(scope: &Scope, lit: T) -> Self {
        T::__expand_as_type(scope).constant(lit.into()).into()
    }

    /// Get the [`ConstantValue`] from the variable.
    pub fn constant(&self) -> Option<ConstantValue> {
        match self.expand.kind {
            VariableKind::Constant(val) => Some(val),
            _ => None,
        }
    }

    pub fn __expand_into_lit_unchecked_method(self, _scope: &Scope) -> T {
        let value = self.constant().unwrap();
        T::from_const_value(value)
    }
}

pub(crate) fn init_mut_expand_element(scope: &Scope, element: &Variable) -> Variable {
    if element.ty.is_ptr() {
        panic!("tried initializing mut for ptr {}", element.ty);
    }
    scope.create_local_mut(element.ty)
}

impl<T: IntoMut> IntoMut for Option<T> {
    fn into_mut(self, scope: &Scope) -> Self {
        self.map(|o| IntoMut::into_mut(o, scope))
    }
}

impl<T: CubeType> CubeType for Vec<T> {
    type ExpandType = Vec<T::ExpandType>;
}

impl<T: IntoExpand> IntoExpand for Vec<T> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<T: ExpandTypeClone> ExpandTypeClone for Vec<T> {
    fn clone_unchecked(&self) -> Self {
        self.iter().map(|it| it.clone_unchecked()).collect()
    }
}

impl<T: IntoMut> IntoMut for Vec<T> {
    fn into_mut(self, scope: &Scope) -> Self {
        self.into_iter().map(|e| e.into_mut(scope)).collect()
    }
}
impl<T: CubeDebug> CubeDebug for Vec<T> {}

impl<T: AsRefExpand> AsRefExpand for Vec<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: AsMutExpand> AsMutExpand for Vec<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

/// Create a constant element of the correct type during expansion.
pub(crate) fn __expand_new<C: Numeric, Out: Numeric>(scope: &Scope, val: C) -> NativeExpand<Out> {
    let input: ConstantValue = val.into();
    Out::__expand_as_type(scope).constant(input).into()
}

impl CubeType for () {
    type ExpandType = ();
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

impl Assign for () {
    fn __expand_assign_method(&mut self, _: &Scope, _: Self) {}
    fn init_mut(&self, _: &Scope) -> Self {}
}

impl IntoRuntime for () {
    fn __expand_runtime_method(self, _: &Scope) -> Self::ExpandType {
        self
    }
}

impl IntoExpand for () {
    type Expand = ();

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl CubeDebug for () {}

impl ExpandTypeClone for () {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl IntoMut for () {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl AsRefExpand for () {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for () {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

pub trait DefaultExpand: CubeType {
    fn __expand_default(scope: &Scope) -> Self::ExpandType;
}

impl<T: CubeType + Default + IntoRuntime> DefaultExpand for T {
    fn __expand_default(scope: &Scope) -> T::ExpandType {
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

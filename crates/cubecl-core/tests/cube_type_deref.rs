#![deny(unused_lifetimes)]

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::ir::{
    AddressType,
    interfaces::TypedExt,
    settings::{Dim3, ExecutionMode, KernelSettings},
};

// Regression for #1891: dereferencing a CubeType struct must read its runtime fields.
#[derive(CubeType, Clone, Copy)]
pub struct Value<F: Float> {
    pub value: F,
}

#[cube]
impl<F: Float> Value<F> {
    pub fn copy(&self) -> Self {
        *self
    }
}

#[derive(CubeType)]
pub struct Wrapper<T: CubeType> {
    pub value: T,
}

#[derive(CubeType)]
pub struct MixedFields {
    // Repeated concrete types must remain valid.
    pub first: f32,
    pub second: f32,
    #[cube(comptime)]
    pub label: String,
}

// Deriving CubeType must remain valid when a concrete field cannot be dereferenced.
#[derive(CubeType)]
pub struct NonDerefField {
    pub values: Box<[f32]>,
}

type ScalarAlias = f32;
type BufferAlias = Box<[f32]>;

#[derive(CubeType, Clone, Copy)]
pub struct AliasedFields {
    pub first: f32,
    pub second: ScalarAlias,
    pub third: core::primitive::f32,
}

#[cube]
impl AliasedFields {
    pub fn copy(&self) -> Self {
        *self
    }
}

#[derive(CubeType)]
pub struct AliasedNonDerefFields {
    pub first: Box<[f32]>,
    pub second: BufferAlias,
}

// The generated bound must not shadow existing lifetimes or require 'static data.
#[derive(CubeType)]
pub struct BorrowedFields<'__cube_deref> {
    pub value: f32,
    #[cube(comptime)]
    pub label: &'__cube_deref str,
}

#[test]
fn deref_reads_mutable_fields() {
    let scope = Scope::root(KernelSettings::new(
        Dim3::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let original = ValueExpand::<f32> {
        value: 2.0f32.into_expand(&scope).into_mut(&scope),
    };
    let copy = original.__expand_copy_method(&scope);

    // Copying a struct must read its fields, rather than alias their mutable storage.
    assert_ne!(original.value.expand, copy.value.expand);
    assert!(original.value.value(&scope).is_ptr(scope.ctx()));
    assert!(!copy.value.value(&scope).is_ptr(scope.ctx()));
}

#[test]
fn deref_nested_structs_and_comptime_fields() {
    let scope = Scope::root(KernelSettings::new(
        Dim3::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let original = WrapperExpand::<MixedFields> {
        value: MixedFieldsExpand {
            first: 3.0f32.into_expand(&scope).into_mut(&scope),
            second: 4.0f32.into_expand(&scope).into_mut(&scope),
            label: String::from("test"),
        },
    };
    let copy = original.__expand_deref_method(&scope);

    assert_eq!(copy.value.label, "test");
    assert!(!copy.value.first.value(&scope).is_ptr(scope.ctx()));
    assert!(!copy.value.second.value(&scope).is_ptr(scope.ctx()));
}

#[test]
fn deref_aliased_fields() {
    let scope = Scope::root(KernelSettings::new(
        Dim3::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let original = AliasedFieldsExpand {
        first: 1.0f32.into_expand(&scope).into_mut(&scope),
        second: 2.0f32.into_expand(&scope).into_mut(&scope),
        third: 3.0f32.into_expand(&scope).into_mut(&scope),
    };
    let copy = original.__expand_copy_method(&scope);

    for (original, copy) in [
        (original.first, copy.first),
        (original.second, copy.second),
        (original.third, copy.third),
    ] {
        assert!(original.value(&scope).is_ptr(scope.ctx()));
        assert!(!copy.value(&scope).is_ptr(scope.ctx()));
        assert_ne!(original.expand, copy.expand);
    }
}

#[test]
fn deref_borrowed_comptime_field() {
    let scope = Scope::root(KernelSettings::new(
        Dim3::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let label = String::from("borrowed");
    let original = BorrowedFieldsExpand {
        value: 1.0f32.into_expand(&scope).into_mut(&scope),
        label: &label,
    };
    let copy = original.__expand_deref_method(&scope);

    assert_eq!(copy.label, label);
    assert!(!copy.value.value(&scope).is_ptr(scope.ctx()));
}

//! A launchable enum's compilation argument is part of the kernel id. Variants
//! compile different kernels, so each must give a different persistent key, even
//! when its fields are runtime values that hash nothing.

use core::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::id::KernelId;

#[allow(dead_code)]
#[derive(CubeType, CubeLaunch)]
enum Shape {
    Empty,
    Other,
    Tuple(u32),
    TupleWide(u32, u32),
    Named { x: u32 },
    NamedWide { x: u32, y: u32 },
}

struct Kernel;

fn hash_of(arg: &ShapeCompilationArg) -> u64 {
    let mut hasher = DefaultHasher::new();
    arg.hash(&mut hasher);
    hasher.finish()
}

fn kernel_key(arg: ShapeCompilationArg) -> impl PartialEq + core::fmt::Debug {
    KernelId::new::<Kernel>().info(arg).stable_hash()
}

fn every_variant() -> [ShapeCompilationArg; 6] {
    [
        ShapeCompilationArg::Empty,
        ShapeCompilationArg::Other,
        ShapeCompilationArg::Tuple(()),
        ShapeCompilationArg::TupleWide((), ()),
        ShapeCompilationArg::Named { x: () },
        ShapeCompilationArg::NamedWide { x: (), y: () },
    ]
}

#[test]
fn every_variant_has_its_own_kernel_key() {
    let variants = every_variant();
    for (i, first) in variants.iter().enumerate() {
        for second in &variants[i + 1..] {
            assert_ne!(first, second);
            assert_ne!(hash_of(first), hash_of(second), "{first:?} and {second:?}");
            assert_ne!(
                kernel_key(first.clone()),
                kernel_key(second.clone()),
                "{first:?} and {second:?} share a persistent kernel key",
            );
        }
    }
}

#[test]
fn equal_arguments_keep_one_kernel_key() {
    for (first, second) in every_variant().into_iter().zip(every_variant()) {
        assert_eq!(hash_of(&first), hash_of(&second));
        assert_eq!(kernel_key(first), kernel_key(second));
    }
}

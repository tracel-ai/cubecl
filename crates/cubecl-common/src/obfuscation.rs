//! Internal home of the [`obfuscate!`](crate::obfuscate) macro. The
//! module itself is private — the macro is exported at the crate root,
//! which is where its documentation lives.

/// Generate a type-erased, inline wrapper for a single value.
///
/// Expands into a private module containing an `Opaque` struct that
/// stores one value of `type:` behind a fixed-shape representation that
/// never names the inner type. Downstream crates that handle the wrapper
/// never have to resolve the inner type, which breaks the transitive
/// monomorphization chain and can visibly cut compile times for deeply
/// generic types.
///
/// # Example
///
/// ```rust
/// use cubecl_common::obfuscate;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Tensor { shape: Vec<usize>, data: Vec<f32> }
///
/// obfuscate!(
///     type: Tensor,
///     module: tensor,
///     align: 8,
///     derives: [Send, Sync, Debug, PartialEq, Clone],
/// );
///
/// # fn main() {
/// let t = tensor::Opaque::new(Tensor { shape: vec![2, 2], data: vec![1.0; 4] });
/// let copy = t.clone();
/// assert_eq!(t, copy);
///
/// // Recover the concrete type when you actually need its fields.
/// let inner: Tensor = t.into_inner();
/// assert_eq!(inner.shape, vec![2, 2]);
/// # }
/// ```
///
/// The macro call must sit at module scope (or crate root); it cannot be
/// nested inside a function body, since the generated module's `super`
/// would not see locals.
///
/// # When to reach for this
///
/// - A type appears in many generic instantiations and shows up as a
///   compile-time bottleneck.
/// - You can't use `Box<dyn Trait>`: you need the concrete type back, or
///   you want to avoid the heap.
/// - The wrapper's API surface (`new` / `as_ref` / `as_mut` / `into_inner`)
///   is enough for the boundary you're crossing.
///
/// # Syntax
///
/// ```text
/// obfuscate!(
///     type:    <inner-type>,
///     module:  <module-name>,
///     align:   <integer-literal>,
///     derives: [<trait>, ...],
/// );
/// ```
///
/// All four keys are mandatory and must appear in that order. `derives:
/// []` is allowed and produces an `Opaque` with no extra impls.
///
/// `align:` must be the integer literal equal to
/// `max(core::mem::align_of::<inner>(), core::mem::align_of::<*mut ()>())`
/// — the inner type's alignment for soundness, clamped up to the
/// pointer alignment because the storage slots are pointer-sized.
/// Wrong values are a `const` assertion failure at the call site, with
/// a message pointing at the right one. A literal is required because
/// `#[repr(align(N))]` does not accept const expressions — the macro
/// can't substitute `align_of::<inner>()` for you.
///
/// # Generated API
///
/// The macro emits a single struct, `Opaque`, with inherent methods
/// scoped `pub(super)` — visible to the parent module of the generated
/// wrapper and nowhere else:
///
/// ```ignore
/// impl Opaque {
///     fn new(inner: Inner) -> Self;
///     fn as_ref(&self) -> &Inner;
///     fn as_mut(&mut self) -> &mut Inner;
///     fn into_inner(self) -> Inner;
/// }
/// // Plus a `Drop` impl that runs the inner's destructor exactly once.
/// // `into_inner` suppresses this drop and returns the value instead.
/// ```
///
/// Any extra trait impls listed in `derives:` are emitted alongside.
///
/// # Derives
///
/// The wrapper is `!Send + !Sync` and has no extra trait impls by
/// default. Opt in via the `derives:` list:
///
/// | Token | Kind | Behaviour |
/// |---|---|---|
/// | `Send`, `Sync` | `unsafe impl` | Caller asserts the inner type satisfies the marker. No bound check is emitted. |
/// | `Debug`, `Display` | safe | Forwards the `Formatter` to the inner value — width/precision/alternate flags survive. |
/// | `PartialEq` | safe | `self.as_ref() == other.as_ref()` |
/// | `Eq` | safe marker | Pairs with `PartialEq`. |
/// | `Hash` | safe | Hashes the inner value. |
/// | `Clone` | safe | Deep clone via the inner's `Clone`. Storage is independent. |
/// | `Default` | safe | Constructs from `<Inner as Default>::default()`. |
///
/// Any other identifier in `derives: [...]` is a compile error at the
/// call site. All non-marker impls require the inner type to implement
/// the trait — the impl is concrete (not generic), so the bound is
/// checked at expansion.
///
/// # Storage and safety
///
/// The wrapper holds the inner value inline — no allocation. Storage
/// uses pointer-sized [`MaybeUninit<*mut ()>`](core::mem::MaybeUninit)
/// slots rather than bytes, so any pointers owned by the inner value
/// retain their provenance through the round trip. A naive
/// `[u8; size_of::<T>()]` buffer would strip pointer provenance on
/// read, breaking under Miri / Tree Borrows for any inner type that
/// owns an `Arc`, `Box`, or similar.
///
/// The wrapper carries `#[repr(C, align(N))]` where `N` is the
/// caller-declared `align:` literal. A `const` assertion verifies the
/// literal matches `max(align_of::<inner>(), align_of::<*mut ()>())`,
/// so wrong values are a compile error rather than runtime UB.
///
/// # Why a macro?
///
/// `size_of::<T>()` and `align_of::<T>()` are usable in `const` position
/// only for concrete types. Without the unstable
/// `feature(generic_const_exprs)`, you cannot write
/// `struct Wrapper<T> { bytes: [u8; size_of::<T>()] }` as a generic. The
/// macro side-steps this by expanding a fresh, concrete wrapper per use
/// site.
#[macro_export]
macro_rules! obfuscate {
    (
        type: $inner:ty,
        module: $mod_name:ident,
        align: $align:literal,
        derives: [ $($trait:ident),* $(,)? ] $(,)?
    ) => {
        mod $mod_name {
            #[allow(unused_imports)]
            use super::*;

            const SIZE: usize = ::core::mem::size_of::<$inner>();
            const SLOT: usize = ::core::mem::size_of::<*mut ()>();

            /// Number of pointer-sized slots needed to cover `$inner`. Round
            /// up; the extra bytes past `SIZE` are never read as part of the
            /// inner value.
            const SLOTS: usize = SIZE.div_ceil(SLOT);

            // Compile-time check that the caller's `align:` literal matches
            // the wrapper's actual alignment requirement. The wrapper is
            // aligned to `max(align_of::<$inner>(), align_of::<*mut ()>())`
            // — the inner type's alignment for soundness, and the pointer
            // alignment as a floor because the storage slots are
            // pointer-sized.
            const _: () = {
                let inner = ::core::mem::align_of::<$inner>();
                let slot = ::core::mem::align_of::<*mut ()>();
                let needed = if inner > slot { inner } else { slot };
                assert!(
                    $align == needed,
                    "obfuscate!: the `align:` literal does not match the required value. \
                     Set it to `max(core::mem::align_of::<YourType>(), core::mem::align_of::<*mut ()>())` \
                     (typically 8 on 64-bit targets, larger if your type has a higher alignment). \
                     A literal is required because `#[repr(align(N))]` does not accept \
                     const expressions like `align_of::<T>()` — the macro can't compute it for you.",
                );
            };

            /// Aligned, opaque storage for one `$inner` value.
            ///
            /// `#[repr(C, align($align))]` raises the struct's alignment to
            /// the caller-declared `$align`, which the const assertion
            /// above guarantees matches `align_of::<$inner>()` (or the
            /// pointer-alignment floor, whichever is larger). That makes
            /// the `*const _ as *const $inner` cast in the methods sound
            /// without naming `$inner` in the struct definition.
            ///
            /// Slots are typed as pointer-sized `MaybeUninit<*mut ()>`
            /// rather than `u8`: the `*mut ()` payload type lets pointers
            /// owned by the inner value retain their provenance through
            /// `ptr::write` → `ptr::read`, while `MaybeUninit` allows the
            /// slots to legally hold uninitialised padding bytes (which
            /// arise for enum types whose variants don't fill the whole
            /// discriminant). Neither wrapper names `$inner`, so the
            /// type-erasure goal is preserved.
            #[repr(C, align($align))]
            pub(super) struct Opaque {
                data: [::core::mem::MaybeUninit<*mut ()>; SLOTS],
            }

            // Opt-in trait impls. Each token in the `derives: [...]` list is
            // dispatched to the helper macro below, which emits exactly that
            // one impl. Unlisted traits are not implemented — `*mut ()`
            // makes `Opaque` `!Send + !Sync` by default.
            $(
                $crate::obfuscate_impl!($trait);
            )*

            // The macro exposes a uniform API (`new`/`as_ref`/`as_mut`/
            // `into_inner`) but not every use site needs every entry point.
            // Silence the resulting `dead_code` warnings here rather than at
            // every call site.
            #[allow(dead_code)]
            impl Opaque {
                /// Wrap an `$inner` value in a fresh `Opaque`.
                pub(super) fn new(inner: $inner) -> Self {
                    let mut wrapper = Self {
                        data: [::core::mem::MaybeUninit::uninit(); SLOTS],
                    };
                    // SAFETY: `data` is at offset 0 of `Self`, and `Self`
                    // has alignment `$align` (the caller-declared literal,
                    // verified by the const assert to match the inner
                    // type's alignment requirement). The cast to
                    // `*mut $inner` is therefore properly aligned. The
                    // write covers exactly `SIZE` bytes, which is
                    // `<= SLOTS * SLOT` bytes of valid, exclusive storage.
                    unsafe {
                        (wrapper.data.as_mut_ptr() as *mut $inner).write(inner);
                    }
                    wrapper
                }

                /// Borrow the wrapped value.
                pub(super) fn as_ref(&self) -> &$inner {
                    // SAFETY: alignment is guaranteed by `repr(align($align))`
                    // combined with the const assertion that matches
                    // `$align` to the inner type's alignment requirement.
                    // The bytes were initialized in `new` and stay
                    // initialized until `Drop`/`into_inner` consume them.
                    unsafe { &*(self.data.as_ptr() as *const $inner) }
                }

                /// Mutably borrow the wrapped value.
                pub(super) fn as_mut(&mut self) -> &mut $inner {
                    // SAFETY: same as `as_ref`; `&mut self` gives exclusive
                    // access.
                    unsafe { &mut *(self.data.as_mut_ptr() as *mut $inner) }
                }

                /// Take ownership of the wrapped value, suppressing `Opaque`'s
                /// `Drop` so the inner value's destructor runs exactly once
                /// (when the returned owner is dropped).
                pub(super) fn into_inner(self) -> $inner {
                    // SAFETY: read the bytes through the correctly-typed
                    // pointer, then forget `self` to skip our `Drop` (which
                    // would otherwise drop the value again).
                    let inner: $inner =
                        unsafe { ::core::ptr::read(self.data.as_ptr() as *const $inner) };
                    ::core::mem::forget(self);
                    inner
                }
            }

            impl ::core::ops::Drop for Opaque {
                fn drop(&mut self) {
                    // SAFETY: see `as_ref`; running once per `Opaque` since
                    // `into_inner` forgets `self` when it consumes the value.
                    unsafe {
                        ::core::ptr::drop_in_place(
                            self.data.as_mut_ptr() as *mut $inner,
                        );
                    }
                }
            }
        }
    };
}

/// Internal helper. Dispatches a single token from [`obfuscate!`]'s
/// `derives: [...]` list to the corresponding impl on the surrounding
/// module's `Opaque` type. Not part of the public API — call
/// [`obfuscate!`] instead.
///
/// Unrecognised tokens produce a "no rules expected" macro error at the
/// call site, which is the intended behaviour for typos (`Sned`) and for
/// derives this crate does not support.
#[macro_export]
#[doc(hidden)]
macro_rules! obfuscate_impl {
    (Send) => {
        // SAFETY: caller of `obfuscate!` asserts that the inner type is
        // safe to move across thread boundaries.
        unsafe impl ::core::marker::Send for Opaque {}
    };
    (Sync) => {
        // SAFETY: caller of `obfuscate!` asserts that the inner type is
        // safe to share across thread boundaries.
        unsafe impl ::core::marker::Sync for Opaque {}
    };
    (Debug) => {
        impl ::core::fmt::Debug for Opaque {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(self.as_ref(), f)
            }
        }
    };
    (Display) => {
        impl ::core::fmt::Display for Opaque {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Display::fmt(self.as_ref(), f)
            }
        }
    };
    (PartialEq) => {
        impl ::core::cmp::PartialEq for Opaque {
            fn eq(&self, other: &Self) -> bool {
                self.as_ref() == other.as_ref()
            }
        }
    };
    (Eq) => {
        impl ::core::cmp::Eq for Opaque {}
    };
    (Hash) => {
        impl ::core::hash::Hash for Opaque {
            fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
                ::core::hash::Hash::hash(self.as_ref(), state)
            }
        }
    };
    (Clone) => {
        impl ::core::clone::Clone for Opaque {
            fn clone(&self) -> Self {
                Self::new(::core::clone::Clone::clone(self.as_ref()))
            }
        }
    };
    (Default) => {
        impl ::core::default::Default for Opaque {
            fn default() -> Self {
                Self::new(::core::default::Default::default())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    // ------------------------------------------------------------------
    // 1. Small plain-old-data round-trip.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Pod {
        a: u64,
        b: u32,
    }

    obfuscate!(
        type: Pod,
        module: pod,
        align: 8,
        derives: [],
    );

    #[test]
    fn pod_round_trip() {
        let value = Pod {
            a: 0xDEAD_BEEF,
            b: 7,
        };
        let wrapper = pod::Opaque::new(value.clone());
        assert_eq!(wrapper.as_ref(), &value);
        assert_eq!(wrapper.into_inner(), value);
    }

    #[test]
    fn pod_mutation() {
        let mut wrapper = pod::Opaque::new(Pod { a: 0, b: 0 });
        wrapper.as_mut().a = 99;
        wrapper.as_mut().b = 1;
        assert_eq!(wrapper.as_ref(), &Pod { a: 99, b: 1 });
    }

    #[test]
    fn opaque_alignment_matches_declared() {
        // The wrapper's alignment is what the caller declared via
        // `align:`, which the macro's const assert verifies matches
        // `max(align_of::<inner>(), align_of::<*mut ()>())`.
        assert_eq!(core::mem::align_of::<pod::Opaque>(), 8);
        assert_eq!(core::mem::align_of::<high_align::Opaque>(), 32);
        assert_eq!(
            core::mem::align_of::<zst::Opaque>(),
            core::mem::align_of::<*mut ()>()
        );
    }

    // ------------------------------------------------------------------
    // 2. Heap-owning type — verifies Drop runs exactly once via `Drop`
    //    AND that `into_inner` does NOT double-drop.
    // ------------------------------------------------------------------

    struct DropCounter(Arc<AtomicUsize>);
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    obfuscate!(
        type: DropCounter,
        module: counted,
        align: 8,
        derives: [],
    );

    #[test]
    fn drop_runs_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let _w = counted::Opaque::new(DropCounter(counter.clone()));
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn into_inner_does_not_double_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let wrapper = counted::Opaque::new(DropCounter(counter.clone()));
        let inner = wrapper.into_inner();
        // The wrapper is gone but the value lives on in `inner`.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        drop(inner);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ------------------------------------------------------------------
    // 3. High-alignment inner type — verifies the static assert is
    //    permissive enough for what we actually use it for.
    // ------------------------------------------------------------------

    #[repr(align(32))]
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct HighAlign {
        data: [u64; 4],
    }

    obfuscate!(
        type: HighAlign,
        module: high_align,
        align: 32,
        derives: [],
    );

    #[test]
    fn high_alignment_inner_works() {
        let value = HighAlign { data: [1, 2, 3, 4] };
        let wrapper = high_align::Opaque::new(value.clone());
        // The wrapper must be exactly the declared alignment; the
        // wrapped reference inherits it.
        assert_eq!(core::mem::align_of::<high_align::Opaque>(), 32);
        let r: &HighAlign = wrapper.as_ref();
        assert_eq!((r as *const HighAlign as usize) % 32, 0);
        assert_eq!(r, &value);
    }

    // ------------------------------------------------------------------
    // 4. Enum with a heap allocation — exercises pointer provenance
    //    preservation through `ptr::write` / `ptr::read`.
    // ------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    #[allow(dead_code)]
    enum Shape {
        Scalar(u64),
        Vector(Vec<u32>),
        Nested(alloc::boxed::Box<Shape>),
    }

    obfuscate!(
        type: Shape,
        module: shape,
        align: 8,
        derives: [],
    );

    #[test]
    fn enum_with_heap_round_trip() {
        let value = Shape::Nested(alloc::boxed::Box::new(Shape::Vector(alloc::vec![
            1, 2, 3, 4
        ])));
        let wrapper = shape::Opaque::new(value.clone());
        let recovered = wrapper.into_inner();
        assert_eq!(recovered, value);
    }

    // ------------------------------------------------------------------
    // 5. Trait forwarding — one module that opts in to every supported
    //    derive, then a test per trait that exercises the impl.
    // ------------------------------------------------------------------

    #[derive(Debug, PartialEq, Eq, Hash, Clone, Default)]
    struct AllTraits {
        x: u64,
        y: String,
    }

    // `Display` has no derive; the macro test relies on this manual impl
    // delegating through the `Opaque`'s `Display` arm.
    impl core::fmt::Display for AllTraits {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "AllTraits(x={}, y={})", self.x, self.y)
        }
    }

    obfuscate!(
        type: AllTraits,
        module: all_traits,
        align: 8,
        derives: [Send, Sync, Debug, Display, PartialEq, Eq, Hash, Clone, Default],
    );

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn opaque_is_send_and_sync_when_listed() {
        assert_send::<all_traits::Opaque>();
        assert_sync::<all_traits::Opaque>();
    }

    #[test]
    fn opaque_debug_forwards_to_inner() {
        let v = AllTraits {
            x: 42,
            y: String::from("hello"),
        };
        let w = all_traits::Opaque::new(v.clone());
        assert_eq!(alloc::format!("{:?}", w), alloc::format!("{:?}", v));
    }

    #[test]
    fn opaque_debug_alternate_format_forwards_to_inner() {
        // `{:#?}` exercises the pretty-printer; the macro forwards the
        // `Formatter` so flags/width/precision survive.
        let v = AllTraits {
            x: 42,
            y: String::from("hello"),
        };
        let w = all_traits::Opaque::new(v.clone());
        assert_eq!(alloc::format!("{:#?}", w), alloc::format!("{:#?}", v));
    }

    #[test]
    fn opaque_display_forwards_to_inner() {
        let v = AllTraits {
            x: 7,
            y: String::from("world"),
        };
        let w = all_traits::Opaque::new(v.clone());
        assert_eq!(alloc::format!("{}", w), alloc::format!("{}", v));
        assert_eq!(alloc::format!("{}", w), "AllTraits(x=7, y=world)");
    }

    #[test]
    fn opaque_eq_marker_when_listed() {
        // `Eq` is a marker — exercise it by demanding the bound at the
        // type level. Compiles iff the macro emitted the impl.
        fn assert_eq_bound<T: Eq>() {}
        assert_eq_bound::<all_traits::Opaque>();
    }

    #[test]
    fn opaque_partial_eq_forwards_to_inner() {
        let a = all_traits::Opaque::new(AllTraits {
            x: 1,
            y: String::from("a"),
        });
        let b = all_traits::Opaque::new(AllTraits {
            x: 1,
            y: String::from("a"),
        });
        let c = all_traits::Opaque::new(AllTraits {
            x: 2,
            y: String::from("a"),
        });
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn opaque_hash_forwards_to_inner() {
        use core::hash::{Hash, Hasher};

        // Tiny deterministic hasher so this stays no_std-friendly.
        struct Fnv(u64);
        impl Hasher for Fnv {
            fn finish(&self) -> u64 {
                self.0
            }
            fn write(&mut self, bytes: &[u8]) {
                for &b in bytes {
                    self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(b as u64);
                }
            }
        }

        let v = AllTraits {
            x: 1,
            y: String::from("a"),
        };
        let w = all_traits::Opaque::new(v.clone());

        let mut h_inner = Fnv(0xcbf29ce484222325);
        v.hash(&mut h_inner);
        let mut h_opaque = Fnv(0xcbf29ce484222325);
        w.hash(&mut h_opaque);

        assert_eq!(h_inner.finish(), h_opaque.finish());
    }

    #[test]
    fn opaque_clone_produces_independent_copy() {
        let original = all_traits::Opaque::new(AllTraits {
            x: 1,
            y: String::from("hi"),
        });
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // Storage is independent — distinct addresses.
        let original_ptr = original.as_ref() as *const _ as usize;
        let cloned_ptr = cloned.as_ref() as *const _ as usize;
        assert_ne!(original_ptr, cloned_ptr);
    }

    #[test]
    fn opaque_clone_deep_copies_owned_heap_data() {
        // Confirm the inner's Clone really ran (not a shallow byte copy):
        // mutating the original's String must not affect the clone.
        let mut original = all_traits::Opaque::new(AllTraits {
            x: 0,
            y: String::from("first"),
        });
        let cloned = original.clone();
        original.as_mut().y = String::from("mutated");
        assert_eq!(cloned.as_ref().y, "first");
    }

    #[test]
    fn opaque_default_constructs_default_inner() {
        let w: all_traits::Opaque = Default::default();
        assert_eq!(w.as_ref(), &AllTraits::default());
    }

    // ------------------------------------------------------------------
    // 6. Zero-sized type — degenerate case.
    // ------------------------------------------------------------------

    #[derive(Debug, PartialEq, Eq, Clone, Default)]
    struct Zst;

    obfuscate!(
        type: Zst,
        module: zst,
        // Zst has alignment 1, but the slot type (`*mut ()`) has
        // pointer alignment, which becomes the floor. Declaring 8
        // honors `max(align_of::<Zst>(), align_of::<*mut ()>())`.
        align: 8,
        derives: [Debug, PartialEq, Eq, Clone, Default],
    );

    #[test]
    fn zero_sized_type_round_trip() {
        let wrapper = zst::Opaque::new(Zst);
        assert_eq!(wrapper.as_ref(), &Zst);
        assert_eq!(wrapper.clone().into_inner(), Zst);
    }
}

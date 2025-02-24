use alloc::borrow::ToOwned;
use core::hash::Hasher;

/// A hash of a type's structure
pub trait TypeHash {
    /// Generate a stable hash of the type structure.
    ///
    /// This recursively hashes the names and types of each variant and field, and uses an unseeded
    /// hasher to ensure the hash is stable across compilations and executions. The hash should only
    /// change if a field/variant is renamed, added, or its type is changed.
    #[allow(unused)]
    fn type_hash() -> u64 {
        let mut hasher = fnv::FnvHasher::default();
        Self::write_hash(&mut hasher);
        hasher.finish()
    }

    /// Write the structure of the type to the hasher
    fn write_hash(hasher: &mut impl Hasher);
}

macro_rules! impl_type_hash {
    ($( $($ty: ident)::* $(<$($l: lifetime,)* $($T: ident $(: $(? $Sized: ident)? $($(+)? $B: ident)*)?),+>)?,)*) => {
        $(
            impl $(<$($l,)* $($T: $crate::TypeHash $($(+ ?$Sized)? $(+ $B)*)? ),*>)? TypeHash for $($ty)::* $(<$($l,)* $($T),+>)? {
                fn write_hash(hasher: &mut impl core::hash::Hasher) {
                    hasher.write(stringify!($($ty)::*).as_bytes());
                    $($(
                        $T::write_hash(hasher);
                    )+)?
                }
            }
        )*
    };
}

impl_type_hash!(
    bool,
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize,
    f32,
    f64,
    str,
    core::any::TypeId,
    alloc::borrow::Cow<'a, T: ?Sized + ToOwned>,
    alloc::boxed::Box<T: ?Sized>,
    core::cell::Cell<T: ?Sized>,
    core::cell::Ref<'a, T: ?Sized>,
    core::cell::RefCell<T: ?Sized>,
    core::cell::RefMut<'a, T>,
    core::cell::UnsafeCell<T>,
    core::cmp::Ordering,
    core::cmp::Reverse<T>,
    alloc::collections::BinaryHeap<T>,
    alloc::collections::BTreeMap<K, V>,
    alloc::collections::BTreeSet<T>,
    alloc::collections::LinkedList<T>,
    alloc::collections::VecDeque<T>,
    core::hash::BuildHasherDefault<T>,
    core::marker::PhantomData<T: ?Sized>,
    core::mem::ManuallyDrop<T: ?Sized>,
    core::mem::MaybeUninit<T>,
    core::net::IpAddr,
    core::net::Ipv4Addr,
    core::net::Ipv6Addr,
    core::net::SocketAddr,
    core::net::SocketAddrV4,
    core::net::SocketAddrV6,
    core::num::FpCategory,
    core::num::NonZeroI128,
    core::num::NonZeroI16,
    core::num::NonZeroI32,
    core::num::NonZeroI64,
    core::num::NonZeroI8,
    core::num::NonZeroIsize,
    core::num::NonZeroU128,
    core::num::NonZeroU16,
    core::num::NonZeroU32,
    core::num::NonZeroU64,
    core::num::NonZeroU8,
    core::num::NonZeroUsize,
    core::num::Wrapping<T>,
    core::ops::Bound<T>,
    core::ops::Range<T>,
    core::ops::RangeFrom<T>,
    core::ops::RangeInclusive<T>,
    core::ops::RangeFull,
    core::ops::RangeTo<T>,
    core::ops::RangeToInclusive<T>,
    core::option::Option<T>,
    core::pin::Pin<T>,
    core::primitive::char,
    core::ptr::NonNull<T: ?Sized>,
    alloc::rc::Rc<T: ?Sized>,
    alloc::rc::Weak<T: ?Sized>,
    core::result::Result<T, E>,
    alloc::string::String,
    core::time::Duration,
    alloc::vec::Vec<T>,
    hashbrown::HashMap<K, V>,
    hashbrown::HashSet<T>,
    portable_atomic::AtomicBool,
    portable_atomic::AtomicI16,
    portable_atomic::AtomicI32,
    portable_atomic::AtomicI64,
    portable_atomic::AtomicI8,
    portable_atomic::AtomicIsize,
    portable_atomic::AtomicPtr<T>,
    portable_atomic::AtomicU16,
    portable_atomic::AtomicU32,
    portable_atomic::AtomicU64,
    portable_atomic::AtomicU8,
    portable_atomic::AtomicUsize,
);

macro_rules! impl_type_hash_tuple {
    ($($T: ident),*) => {
        impl <$($T: $crate::TypeHash),*> TypeHash for ($($T,)*) {
            fn write_hash(hasher: &mut impl core::hash::Hasher) {
                hasher.write(b"()");
                $(
                    $T::write_hash(hasher);
                )*
            }
        }
    };
}

variadics_please::all_tuples!(impl_type_hash_tuple, 0, 16, T);

impl<T: TypeHash, const N: usize> TypeHash for [T; N] {
    fn write_hash(hasher: &mut impl core::hash::Hasher) {
        hasher.write(b"[;]");
        hasher.write_usize(N);
        T::write_hash(hasher);
    }
}

impl<T: TypeHash + ?Sized> TypeHash for *const T {
    fn write_hash(hasher: &mut impl Hasher) {
        hasher.write(b"*const");
        T::write_hash(hasher);
    }
}

impl<T: TypeHash + ?Sized> TypeHash for *mut T {
    fn write_hash(hasher: &mut impl Hasher) {
        hasher.write(b"*mut");
        T::write_hash(hasher);
    }
}

impl<T: TypeHash> TypeHash for [T] {
    fn write_hash(hasher: &mut impl Hasher) {
        hasher.write(b"[]");
        T::write_hash(hasher);
    }
}

impl<T: TypeHash + ?Sized> TypeHash for &T {
    fn write_hash(hasher: &mut impl Hasher) {
        hasher.write(b"&");
        T::write_hash(hasher);
    }
}

impl<T: TypeHash + ?Sized> TypeHash for &mut T {
    fn write_hash(hasher: &mut impl Hasher) {
        hasher.write(b"&mut");
        T::write_hash(hasher);
    }
}

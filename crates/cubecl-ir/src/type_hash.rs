use std::borrow::ToOwned;
use std::hash::Hasher;

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
                fn write_hash(hasher: &mut impl std::hash::Hasher) {
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
    std::any::TypeId,
    std::borrow::Cow<'a, T: ?Sized + ToOwned>,
    std::boxed::Box<T: ?Sized>,
    std::cell::Cell<T: ?Sized>,
    std::cell::Ref<'a, T: ?Sized>,
    std::cell::RefCell<T: ?Sized>,
    std::cell::RefMut<'a, T>,
    std::cell::UnsafeCell<T>,
    std::cmp::Ordering,
    std::cmp::Reverse<T>,
    std::collections::BinaryHeap<T>,
    std::collections::BTreeMap<K, V>,
    std::collections::BTreeSet<T>,
    std::collections::HashMap<K, V>,
    std::collections::HashSet<T>,
    std::collections::LinkedList<T>,
    std::collections::VecDeque<T>,
    std::ffi::c_void,
    std::ffi::CStr,
    std::ffi::CString,
    std::ffi::OsStr,
    std::ffi::OsString,
    std::hash::BuildHasherDefault<T>,
    std::marker::PhantomData<T: ?Sized>,
    std::mem::ManuallyDrop<T: ?Sized>,
    std::mem::MaybeUninit<T>,
    std::net::IpAddr,
    std::net::Ipv4Addr,
    std::net::Ipv6Addr,
    std::net::SocketAddr,
    std::net::SocketAddrV4,
    std::net::SocketAddrV6,
    std::num::FpCategory,
    std::num::NonZeroI128,
    std::num::NonZeroI16,
    std::num::NonZeroI32,
    std::num::NonZeroI64,
    std::num::NonZeroI8,
    std::num::NonZeroIsize,
    std::num::NonZeroU128,
    std::num::NonZeroU16,
    std::num::NonZeroU32,
    std::num::NonZeroU64,
    std::num::NonZeroU8,
    std::num::NonZeroUsize,
    std::num::Wrapping<T>,
    std::ops::Bound<T>,
    std::ops::Range<T>,
    std::ops::RangeFrom<T>,
    std::ops::RangeInclusive<T>,
    std::ops::RangeFull,
    std::ops::RangeTo<T>,
    std::ops::RangeToInclusive<T>,
    std::option::Option<T>,
    std::path::Path,
    std::path::PathBuf,
    std::pin::Pin<T>,
    std::primitive::char,
    std::ptr::NonNull<T: ?Sized>,
    std::rc::Rc<T: ?Sized>,
    std::rc::Weak<T: ?Sized>,
    std::result::Result<T, E>,
    std::string::String,
    std::sync::atomic::AtomicBool,
    std::sync::atomic::AtomicI16,
    std::sync::atomic::AtomicI32,
    std::sync::atomic::AtomicI64,
    std::sync::atomic::AtomicI8,
    std::sync::atomic::AtomicIsize,
    std::sync::atomic::AtomicPtr<T>,
    std::sync::atomic::AtomicU16,
    std::sync::atomic::AtomicU32,
    std::sync::atomic::AtomicU64,
    std::sync::atomic::AtomicU8,
    std::sync::atomic::AtomicUsize,
    std::sync::mpsc::Receiver<T>,
    std::sync::mpsc::Sender<T>,
    std::sync::mpsc::SyncSender<T>,
    std::sync::Arc<T: ?Sized>,
    std::sync::Mutex<T: ?Sized>,
    std::sync::Once,
    std::sync::RwLock<T: ?Sized>,
    std::sync::RwLockReadGuard<'a, T: ?Sized>,
    std::sync::RwLockWriteGuard<'a, T: ?Sized>,
    std::sync::Weak<T: ?Sized>,
    std::thread::Builder,
    std::thread::JoinHandle<T>,
    std::thread::LocalKey<T>,
    std::thread::Thread,
    std::thread::ThreadId,
    std::time::Duration,
    std::time::Instant,
    std::time::SystemTime,
    std::vec::Vec<T>,
);

macro_rules! impl_type_hash_tuple {
    ($($T: ident),*) => {
        impl <$($T: $crate::TypeHash),*> TypeHash for ($($T,)*) {
            fn write_hash(hasher: &mut impl std::hash::Hasher) {
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
    fn write_hash(hasher: &mut impl std::hash::Hasher) {
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

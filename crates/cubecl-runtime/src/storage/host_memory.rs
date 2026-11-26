use crate::server::IoError;
use alloc::alloc::{Layout, alloc_zeroed, dealloc};
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;
use core::any::TypeId;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Range;
use core::ptr::NonNull;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::bytes::{
    AccessError, AccessPolicy, AllocationController, AllocationProperty, Bytes,
};
use cubecl_environment::sync::Mutex;

/// The alignment of [`HostMemory`] and of every range imported from it: the largest page a
/// runtime imports host memory at (Metal's 16 KiB, Vulkan's `minImportedHostPointerAlignment`),
/// rounded up.
pub const HOST_MEMORY_ALIGNMENT: usize = 64 * 1024;

/// Zeroed host memory that the kernels of one or more devices read in place once each device
/// imports a range of it
/// ([`Client::import_host_memory`](crate::client::Client::import_host_memory)), with no copy to
/// the device.
///
/// Written through [`bytes_mut`](Self::bytes_mut) while one owner holds it, then shared behind an
/// [`Arc`](alloc::sync::Arc) and read-only: a kernel reads it while the host could otherwise
/// write it, and nothing orders the two.
///
/// # Guarantees
///
/// The memory is freed only once every runtime that registered it has released its registration,
/// which can wait for that runtime's devices. A registration that cannot be released leaks the
/// memory instead: the runtime's devices may still read it.
///
/// # Examples
///
/// ```
/// use cubecl_runtime::storage::{HOST_MEMORY_ALIGNMENT, HostMemory};
///
/// let mut memory = HostMemory::new(3 * HOST_MEMORY_ALIGNMENT).unwrap();
/// memory.bytes_mut()[0] = 7;
/// assert_eq!(memory.bytes()[0], 7);
/// assert_eq!(memory.as_ptr() as usize % HOST_MEMORY_ALIGNMENT, 0);
/// ```
pub struct HostMemory {
    /// Membership by the registration's type; released in arrival order when the memory drops.
    registrations: Mutex<Vec<(TypeId, Box<dyn HostReleaser>)>>,
    bytes: ManuallyDrop<Bytes>,
}

impl HostMemory {
    /// `size_bytes` of zeroed host memory, its start and its size rounded up to
    /// [`HOST_MEMORY_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// [`IoError::HostMemoryRefused`] when the size is zero or the host cannot allocate it.
    pub fn new(size_bytes: usize) -> Result<Self, IoError> {
        let refused = |reason: &str| IoError::HostMemoryRefused {
            reason: format!("allocating {size_bytes} bytes: {reason}").into(),
            backtrace: BackTrace::capture(),
        };
        if size_bytes == 0 {
            return Err(refused("an empty allocation imports nothing"));
        }
        let layout = size_bytes
            .checked_next_multiple_of(HOST_MEMORY_ALIGNMENT)
            .and_then(|size| Layout::from_size_align(size, HOST_MEMORY_ALIGNMENT).ok())
            .ok_or_else(|| refused("past the largest allocation the host addresses"))?;
        // SAFETY: the layout's size is not zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| refused("the host is out of memory"))?;
        let controller = HostAllocationController { ptr, layout };
        // SAFETY: the allocation holds `layout.size()` zeroed bytes.
        let bytes = unsafe { Bytes::from_controller(Box::new(controller), layout.size()) };
        Ok(Self {
            registrations: Mutex::new(Vec::new()),
            bytes: ManuallyDrop::new(bytes),
        })
    }

    /// The bytes held, a multiple of [`HOST_MEMORY_ALIGNMENT`].
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Always false: an empty [`HostMemory`] is refused.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The first byte, the address a runtime registers.
    pub fn as_ptr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    /// The bytes held.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The bytes held, for the one owner to write before the memory is shared.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Whether a device may import `range`: non-empty, inside the memory and aligned to
    /// [`HOST_MEMORY_ALIGNMENT`].
    ///
    /// # Errors
    ///
    /// [`IoError::HostMemoryRefused`] for any other range.
    pub fn ensure_importable(&self, range: &Range<u64>) -> Result<(), IoError> {
        let alignment = HOST_MEMORY_ALIGNMENT as u64;
        let importable = !range.is_empty()
            && range.end <= self.len() as u64
            && range.start.is_multiple_of(alignment)
            && range.end.is_multiple_of(alignment);
        match importable {
            true => Ok(()),
            false => Err(IoError::HostMemoryRefused {
                reason: format!(
                    "range {range:?} of {} bytes: it must be non-empty, inside and aligned to {alignment}",
                    self.len()
                )
                .into(),
                backtrace: BackTrace::capture(),
            }),
        }
    }

    /// Run `register` unless a registration of type `R` already exists, and keep what it returns
    /// until the memory is dropped.
    ///
    /// For a runtime whose registration covers every device at once, such as one pinning for all
    /// of them: the first device to import registers, the others find it.
    ///
    /// # Assumptions
    ///
    /// `register` registers nothing on this memory itself: it runs under this memory's lock.
    ///
    /// # Errors
    ///
    /// Whatever `register` returns; nothing is kept then, and the next call runs it again.
    pub fn register_once<R: HostReleaser>(
        &self,
        register: impl FnOnce(&Self) -> Result<R, IoError>,
    ) -> Result<(), IoError> {
        let mut registrations = self.registrations.lock();
        if registrations.iter().any(|(id, _)| *id == TypeId::of::<R>()) {
            return Ok(());
        }
        let registration = register(self)?;
        registrations.push((TypeId::of::<R>(), Box::new(registration)));
        Ok(())
    }
}

/// A runtime's registration of a [`HostMemory`], which it releases before the memory is freed.
///
/// # Assumptions
///
/// [`release`](Self::release) is called once, when the memory drops, and the registration is
/// dropped right after it, before the memory is freed.
///
/// # Guarantees
///
/// Once [`release`](Self::release) returns true, no device of the runtime reads the memory.
pub trait HostReleaser: Send + Sync + 'static {
    /// Release the registration. False when it could not be released: the runtime's devices may
    /// still read the memory, so it is never freed.
    fn release(&mut self) -> bool;
}

impl Drop for HostMemory {
    fn drop(&mut self) {
        let held = self
            .registrations
            .lock()
            .drain(..)
            .map(|(_, mut registration)| registration.release())
            .filter(|released| !released)
            .count();
        if held > 0 {
            return;
        }
        // SAFETY: dropped once, here, and never read again.
        unsafe { ManuallyDrop::drop(&mut self.bytes) };
    }
}

impl core::fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostMemory")
            .field("ptr", &self.as_ptr())
            .field("len", &self.len())
            .finish()
    }
}

/// Owns the aligned allocation behind a [`HostMemory`]'s bytes, freed when they drop.
struct HostAllocationController {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AllocationController for HostAllocationController {
    fn alloc_align(&self) -> usize {
        self.layout.align()
    }

    fn property(&self) -> AllocationProperty {
        AllocationProperty::Native
    }

    fn may_alias_mutable_storage(&self) -> bool {
        false
    }

    unsafe fn memory_mut(
        &mut self,
        _policy: AccessPolicy,
    ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
        // SAFETY: `ptr` holds `layout.size()` bytes for as long as `self` lives, and `&mut self`
        // excludes every other reader.
        Ok(
            unsafe {
                core::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.layout.size())
            },
        )
    }

    fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
        // SAFETY: `ptr` holds `layout.size()` bytes for as long as `self` lives.
        Ok(unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast(), self.layout.size()) })
    }
}

impl Drop for HostAllocationController {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with `layout` and is freed once, here.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn memory_is_zeroed() {
        let memory = HostMemory::new(1).unwrap();
        assert!(memory.bytes().iter().all(|&byte| byte == 0));
    }

    #[test]
    fn memory_starts_aligned() {
        let memory = HostMemory::new(1).unwrap();
        assert_eq!(memory.as_ptr() as usize % HOST_MEMORY_ALIGNMENT, 0);
    }

    #[test]
    fn memory_size_rounds_up_to_the_alignment() {
        let memory = HostMemory::new(HOST_MEMORY_ALIGNMENT + 1).unwrap();
        assert_eq!(memory.len(), 2 * HOST_MEMORY_ALIGNMENT);
    }

    #[test]
    fn an_allocation_past_the_address_space_is_refused() {
        assert!(matches!(
            HostMemory::new(usize::MAX),
            Err(IoError::HostMemoryRefused { .. })
        ));
    }

    #[test]
    fn an_empty_allocation_is_refused() {
        assert!(matches!(
            HostMemory::new(0),
            Err(IoError::HostMemoryRefused { .. })
        ));
    }

    #[test]
    fn an_aligned_range_inside_is_importable() {
        let memory = HostMemory::new(2 * HOST_MEMORY_ALIGNMENT).unwrap();
        let chunk = HOST_MEMORY_ALIGNMENT as u64;
        assert!(memory.ensure_importable(&(chunk..2 * chunk)).is_ok());
    }

    #[test]
    fn a_range_past_the_end_is_not_importable() {
        let memory = HostMemory::new(HOST_MEMORY_ALIGNMENT).unwrap();
        let chunk = HOST_MEMORY_ALIGNMENT as u64;
        assert!(memory.ensure_importable(&(0..2 * chunk)).is_err());
    }

    #[test]
    fn an_unaligned_range_is_not_importable() {
        let memory = HostMemory::new(HOST_MEMORY_ALIGNMENT).unwrap();
        assert!(memory.ensure_importable(&(0..1)).is_err());
    }

    #[test]
    fn an_empty_range_is_not_importable() {
        let memory = HostMemory::new(HOST_MEMORY_ALIGNMENT).unwrap();
        assert!(memory.ensure_importable(&(0..0)).is_err());
    }

    struct Counted(Arc<AtomicUsize>);

    impl HostReleaser for Counted {
        fn release(&mut self) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst);
            true
        }
    }

    struct Unit;

    impl HostReleaser for Unit {
        fn release(&mut self) -> bool {
            true
        }
    }

    #[test]
    fn a_registration_runs_once() {
        let runs = AtomicUsize::new(0);
        let memory = HostMemory::new(1).unwrap();
        for _ in 0..3 {
            memory
                .register_once(|_| {
                    runs.fetch_add(1, Ordering::SeqCst);
                    Ok(Unit)
                })
                .unwrap();
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_registration_is_released_on_drop() {
        let released = Arc::new(AtomicUsize::new(0));
        let memory = HostMemory::new(1).unwrap();
        memory
            .register_once(|_| Ok(Counted(released.clone())))
            .unwrap();
        drop(memory);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    struct Held;

    impl HostReleaser for Held {
        fn release(&mut self) -> bool {
            false
        }
    }

    #[test]
    fn a_held_registration_still_releases_the_others() {
        let released = Arc::new(AtomicUsize::new(0));
        let memory = HostMemory::new(1).unwrap();
        memory.register_once(|_| Ok(Held)).unwrap();
        memory
            .register_once(|_| Ok(Counted(released.clone())))
            .unwrap();
        drop(memory);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failed_registration_is_tried_again() {
        let memory = HostMemory::new(1).unwrap();
        let refused = memory.register_once::<Unit>(|_| {
            Err(IoError::HostMemoryRefused {
                reason: "refused".into(),
                backtrace: BackTrace::capture(),
            })
        });
        assert!(refused.is_err());
        memory.register_once(|_| Ok(Unit)).unwrap();
    }
}

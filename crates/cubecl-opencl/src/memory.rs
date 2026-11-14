//! Device memory: pages whose every byte is an address a kernel takes as it is, whatever its offset. Intel's unified
//! shared memory (device allocations) where the device has it, else coarse-grained shared virtual memory.

use crate::compute::cl::{self, ClError, Event, api};
use crate::device::Info;
use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use opencl_sys::{
    CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, CL_DEVICE_PLATFORM, CL_FALSE, CL_MEM_READ_WRITE,
    CL_SUCCESS, CL_TRUE, CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL, cl_bool, cl_command_queue,
    cl_context, cl_device_id, cl_event, cl_int, cl_kernel, cl_mem_properties_intel, cl_platform_id,
    cl_uint, clDeviceMemAllocINTEL_t, clEnqueueMemcpyINTEL_t, clMemFreeINTEL_t,
    clSetKernelArgMemPointerINTEL_t,
};
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::{mem, ptr};

/// How a device's pages are allocated, handed to kernels and copied.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Memory {
    /// Coarse-grained shared virtual memory.
    Svm,
    /// Device allocations of `cl_intel_unified_shared_memory`: the device's own memory, which integrated GPUs stream
    /// and cache faster than shared virtual memory (mapped as the host's).
    Usm(Usm),
}

/// The entry points of `cl_intel_unified_shared_memory`, from the platform (loaders do not export them).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Usm {
    device: cl_device_id,
    alloc: unsafe extern "C" fn(
        cl_context,
        cl_device_id,
        *const cl_mem_properties_intel,
        usize,
        cl_uint,
        *mut cl_int,
    ) -> *mut c_void,
    free: unsafe extern "C" fn(cl_context, *mut c_void) -> cl_int,
    set_arg: unsafe extern "C" fn(cl_kernel, cl_uint, *const c_void) -> cl_int,
    copy: unsafe extern "C" fn(
        cl_command_queue,
        cl_bool,
        *mut c_void,
        *const c_void,
        usize,
        cl_uint,
        *const cl_event,
        *mut cl_event,
    ) -> cl_int,
}

// SAFETY: a device handle and entry points of the platform's thread-safe API.
unsafe impl Send for Usm {}
// SAFETY: as above.
unsafe impl Sync for Usm {}

impl Memory {
    /// Unified shared memory where the device allocates it (the extension alone may cover host or shared
    /// allocations only) and its platform has the entry points, else shared virtual memory.
    pub(crate) fn of(device: cl_device_id, info: &Info) -> Self {
        let device_allocations = info.u64(CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL)
            & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL
            != 0;
        if !info.extensions(&["cl_intel_unified_shared_memory"]) || !device_allocations {
            return Self::Svm;
        }
        let platform = info.u64(CL_DEVICE_PLATFORM) as usize as cl_platform_id;
        // null where the platform lacks it (None once an optional function pointer)
        let entry = |name: &CStr| {
            api()
                .clGetExtensionFunctionAddressForPlatform(platform, name.as_ptr())
                .unwrap_or(ptr::null_mut())
        };
        // SAFETY: the platform's entry points of these names, which have the extension's signatures.
        let usm = (|| unsafe {
            Some(Usm {
                device,
                alloc: mem::transmute::<*mut c_void, clDeviceMemAllocINTEL_t>(entry(
                    c"clDeviceMemAllocINTEL",
                ))?,
                free: mem::transmute::<*mut c_void, clMemFreeINTEL_t>(entry(c"clMemFreeINTEL"))?,
                set_arg: mem::transmute::<*mut c_void, clSetKernelArgMemPointerINTEL_t>(entry(
                    c"clSetKernelArgMemPointerINTEL",
                ))?,
                copy: mem::transmute::<*mut c_void, clEnqueueMemcpyINTEL_t>(entry(
                    c"clEnqueueMemcpyINTEL",
                ))?,
            })
        })();
        usm.map_or(Self::Svm, Self::Usm)
    }

    /// A page of `size` bytes aligned to `alignment`, or null: the device had no room for it.
    fn alloc(&self, context: cl_context, size: usize, alignment: u32) -> *mut c_void {
        match self {
            Self::Svm => api()
                .clSVMAlloc(context, CL_MEM_READ_WRITE, size, alignment)
                .unwrap_or(ptr::null_mut()),
            Self::Usm(usm) => {
                let mut status = CL_SUCCESS;
                // SAFETY: a live context and its device, no properties.
                unsafe {
                    (usm.alloc)(
                        context,
                        usm.device,
                        ptr::null(),
                        size,
                        alignment,
                        &mut status,
                    )
                }
            }
        }
    }

    /// Frees a page nothing uses any more.
    fn free(&self, context: cl_context, page: *mut c_void) {
        match self {
            Self::Svm => _ = api().clSVMFree(context, page),
            // SAFETY: a page of this context.
            Self::Usm(usm) => _ = unsafe { (usm.free)(context, page) },
        }
    }

    /// Sets argument `index` of `kernel` to the device address `ptr`.
    ///
    /// # Safety
    ///
    /// `kernel` is live and `ptr` lies in this memory, in the kernel's context.
    pub(crate) unsafe fn set_arg(
        &self,
        kernel: cl_kernel,
        index: u32,
        ptr: *const c_void,
    ) -> Result<(), ClError> {
        match self {
            Self::Svm => cl::status(
                api().clSetKernelArgSVMPointer(kernel, index, ptr),
                "clSetKernelArgSVMPointer",
            ),
            // SAFETY: as the function's.
            Self::Usm(usm) => cl::status(
                Some(unsafe { (usm.set_arg)(kernel, index, ptr) }),
                "clSetKernelArgMemPointerINTEL",
            ),
        }
    }

    /// Copies `size` bytes between this memory and the host (or within either).
    ///
    /// # Safety
    ///
    /// `queue` is live, both ranges are valid for `size` bytes, and a non-blocking copy's ranges outlive the
    /// copy (until the queue has run it).
    pub(crate) unsafe fn copy(
        &self,
        queue: cl_command_queue,
        blocking: bool,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> Result<(), ClError> {
        if size == 0 {
            return Ok(());
        }
        let blocking = if blocking { CL_TRUE } else { CL_FALSE };
        let (none, no_event) = (ptr::null(), ptr::null_mut());
        match self {
            Self::Svm => cl::status(
                api().clEnqueueSVMMemcpy(queue, blocking, dst, src, size, 0, none, no_event),
                "clEnqueueSVMMemcpy",
            ),
            // SAFETY: as the function's.
            Self::Usm(usm) => cl::status(
                Some(unsafe { (usm.copy)(queue, blocking, dst, src, size, 0, none, no_event) }),
                "clEnqueueMemcpyINTEL",
            ),
        }
    }
}

/// A slice of device memory: its first byte and its length.
#[derive(Debug, Clone, Copy)]
pub struct OpenClResource {
    pub(crate) ptr: *mut u8,
    pub(crate) size: u64,
}

// SAFETY: a device address, only dereferenced by the device; the stream's queue orders every access to it.
unsafe impl Send for OpenClResource {}
// SAFETY: as above.
unsafe impl Sync for OpenClResource {}

impl OpenClResource {
    /// The address `offset` bytes in (the caller keeps it in the page).
    pub(crate) fn at(&self, offset: usize) -> *mut c_void {
        self.ptr.wrapping_add(offset).cast()
    }
}

/// The pages of one stream: allocated in its context, freed behind its queue's work.
#[derive(Debug)]
pub struct OpenClStorage {
    pages: HashMap<StorageId, (*mut c_void, u64)>,
    /// Pages released while the queue may still use them, freed once the marker after them completes.
    released: Vec<(*mut c_void, Event)>,
    allocated: u64,
    memory: Memory,
    context: cl_context,
    queue: cl_command_queue,
    alignment: usize,
    max_page: u64,
}

// SAFETY: the context and queue are thread-safe OpenCL handles this storage holds references to, and the pages
// are device memory only the queue's commands touch; the server owns the storage.
unsafe impl Send for OpenClStorage {}

impl OpenClStorage {
    /// The storage of the stream on `queue` (in `context`), with pages of `memory` aligned to `alignment` bytes and
    /// at most `max_page` bytes each; holds references to both handles.
    pub(crate) fn new(
        memory: Memory,
        context: cl_context,
        queue: cl_command_queue,
        alignment: usize,
        max_page: u64,
    ) -> Self {
        api().clRetainContext(context);
        api().clRetainCommandQueue(queue);
        Self {
            pages: HashMap::new(),
            released: Vec::new(),
            allocated: 0,
            memory,
            context,
            queue,
            alignment,
            max_page,
        }
    }

    /// Frees every released page once the queue is done with them: submits the queue, then waits for the
    /// marker after the newest (each marker completes after everything enqueued before it, older markers
    /// included).
    fn free_all_released(&mut self) {
        let Some((_, newest)) = self.released.last() else {
            return;
        };
        let waited = cl::flush(self.queue).and_then(|()| newest.wait());
        if let Err(err) = waited {
            // The queue is unusable: wait for all of it instead.
            log::warn!("{err}; freeing after a finish");
            api().clFinish(self.queue);
        }
        for (page, _) in self.released.drain(..) {
            self.memory.free(self.context, page);
        }
    }

    /// Frees the released pages the queue is done with.
    fn free_released(&mut self) {
        let (memory, context) = (self.memory, self.context);
        self.released.retain(|(page, freed)| {
            let done = freed.done();
            if done {
                memory.free(context, *page);
            }
            !done
        });
    }
}

impl Drop for OpenClStorage {
    fn drop(&mut self) {
        // Every command using the pages has run once the queue is finished.
        api().clFinish(self.queue);
        let pages = self.pages.values().map(|(page, _)| *page);
        for page in pages.chain(self.released.drain(..).map(|(page, _)| page)) {
            self.memory.free(self.context, page);
        }
        api().clReleaseCommandQueue(self.queue);
        api().clReleaseContext(self.context);
    }
}

impl ComputeStorage for OpenClStorage {
    type Resource = OpenClResource;

    fn alignment(&self) -> usize {
        self.alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        let (ptr, size) =
            self.pages
                .get(&handle.id)
                .ok_or_else(|| IoError::StorageHandleNotFound {
                    reason: format!("{} in the OpenCL storage", handle.id).into(),
                    backtrace: BackTrace::capture(),
                })?;
        if handle.offset() + handle.size() > *size {
            return Err(IoError::Unknown {
                description: format!(
                    "slice {}..{} out of its {size}-byte page",
                    handle.offset(),
                    handle.offset() + handle.size()
                ),
                backtrace: BackTrace::capture(),
            });
        }
        Ok(OpenClResource {
            ptr: ptr.cast::<u8>().wrapping_add(handle.offset() as usize),
            size: handle.size(),
        })
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        if size > self.max_page {
            return Err(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            });
        }
        self.free_released();
        let size_bytes = size.max(1) as usize;
        let mut ptr = self
            .memory
            .alloc(self.context, size_bytes, self.alignment as u32);
        if ptr.is_null() && !self.released.is_empty() {
            // Room the pages released behind the queue may give back once it has run.
            self.free_all_released();
            ptr = self
                .memory
                .alloc(self.context, size_bytes, self.alignment as u32);
        }
        if ptr.is_null() {
            // the allocation reports no reason: the device had no room for it now
            return Err(IoError::OutOfMemory {
                size,
                backtrace: BackTrace::capture(),
            });
        }
        let id = StorageId::new();
        self.allocated += size;
        self.pages.insert(id, (ptr, size));
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        let Some((page, size)) = self.pages.remove(&id) else {
            return;
        };
        self.allocated -= size;
        self.free_released();
        // Freed once the commands enqueued before it have run.
        // SAFETY: the storage's own queue.
        match unsafe { cl::marker(self.queue) } {
            Ok(freed) => self.released.push((page, freed)),
            Err(err) => {
                // The queue is unusable: wait for it, then free directly.
                log::warn!("{err}; freeing after a finish");
                api().clFinish(self.queue);
                self.memory.free(self.context, page);
            }
        }
    }

    /// Frees the pages released so far, waiting for the queue's work on them: what an explicit cleanup
    /// asks for, as CUDA's synchronous frees do.
    fn flush(&mut self) {
        self.free_all_released();
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}

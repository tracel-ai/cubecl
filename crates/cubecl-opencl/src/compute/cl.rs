//! The `OpenCL` calls of the hot paths, through the library `cl3` loaded at run time.

use cubecl_core::server::{IoError, LaunchError, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::driver::DevicePoison;
use opencl_sys::{
    CL_COMPLETE, CL_EVENT_COMMAND_EXECUTION_STATUS, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_OUT_OF_HOST_MEMORY, CL_OUT_OF_RESOURCES, CL_SUCCESS,
    cl_command_queue, cl_event, cl_int, cl_kernel, cl_profiling_info, cl_uint,
};
use std::ptr;

/// The loaded `OpenCL` library.
pub(crate) fn api() -> &'static cl3::OpenClRuntime {
    cl3::load_library()
        .as_ref()
        .expect("the OpenCL library should be loaded before any device is used")
}

/// A failed `OpenCL` call: its name and status (`None`: the library lacks the entry point).
#[derive(Debug, Clone)]
pub struct ClError {
    /// The entry point called.
    pub call: &'static str,
    /// The status it returned, `None` where the library lacks the entry point.
    pub code: Option<cl_int>,
}

impl core::fmt::Display for ClError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.code {
            Some(code) => {
                let text = cl3::error_codes::error_text(code);
                write!(f, "{} failed ({code}: {text})", self.call)
            }
            None => write!(f, "{} is missing from the OpenCL library", self.call),
        }
    }
}

impl From<ClError> for ServerError {
    fn from(err: ClError) -> Self {
        ServerError::Generic {
            reason: err.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<ClError> for IoError {
    fn from(err: ClError) -> Self {
        IoError::Unknown {
            description: err.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

impl From<ClError> for LaunchError {
    fn from(err: ClError) -> Self {
        let reason = err.to_string();
        let backtrace = BackTrace::capture();
        match err.code {
            Some(
                CL_MEM_OBJECT_ALLOCATION_FAILURE | CL_OUT_OF_RESOURCES | CL_OUT_OF_HOST_MEMORY,
            ) => LaunchError::OutOfMemory { reason, backtrace },
            _ => LaunchError::Unknown { reason, backtrace },
        }
    }
}

impl ClError {
    /// The error of a `cl3` call `call` that returned `code`.
    pub(crate) fn of(call: &'static str) -> impl FnOnce(cl_int) -> Self {
        move |code| Self {
            call,
            code: Some(code),
        }
    }

    /// A failed wait for the device's work. A command that failed, or `CL_OUT_OF_RESOURCES` (what
    /// a driver answers once it has given up on the device, after a hang or a page fault), poisons
    /// the device.
    pub fn into_sync(self) -> ServerError {
        match self.code {
            Some(CL_OUT_OF_RESOURCES | CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) => {
                DevicePoison::new(self.to_string()).into()
            }
            _ => self.into(),
        }
    }
}

/// `Ok` where `result` is `CL_SUCCESS`, else the error of `call`.
pub(crate) fn status(result: Option<cl_int>, call: &'static str) -> Result<(), ClError> {
    match result {
        Some(CL_SUCCESS) => Ok(()),
        code => Err(ClError { call, code }),
    }
}

/// An event this side owns one reference to.
#[derive(Debug)]
pub struct Event(cl_event);

// SAFETY: OpenCL events are thread-safe handles, and this one owns its reference.
unsafe impl Send for Event {}
// SAFETY: as above; `&Event` only queries or waits.
unsafe impl Sync for Event {}

impl Drop for Event {
    fn drop(&mut self) {
        // The reference this value owns.
        api().clReleaseEvent(self.0);
    }
}

impl Clone for Event {
    fn clone(&self) -> Self {
        // A reference of the clone's own.
        api().clRetainEvent(self.0);
        Self(self.0)
    }
}

impl Event {
    /// Waits for the command to complete.
    pub fn wait(&self) -> Result<(), ClError> {
        status(api().clWaitForEvents(1, &self.0), "clWaitForEvents")
    }

    /// Whether the command has completed (or failed: a negative status).
    pub fn done(&self) -> bool {
        let mut value: cl_int = 1;
        let queried = api().clGetEventInfo(
            self.0,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            size_of::<cl_int>(),
            (&mut value as *mut cl_int).cast(),
            ptr::null_mut(),
        );
        queried == Some(CL_SUCCESS) && value <= CL_COMPLETE as cl_int
    }

    /// A profiling timestamp of the command (device clock, ns): its queue profiles.
    pub fn time_ns(&self, param: cl_profiling_info) -> Result<u64, ClError> {
        let mut value = 0u64;
        status(
            api().clGetEventProfilingInfo(
                self.0,
                param,
                size_of::<u64>(),
                (&mut value as *mut u64).cast(),
                ptr::null_mut(),
            ),
            "clGetEventProfilingInfo",
        )?;
        Ok(value)
    }
}

/// Enqueues `kernel` over `groups` work-groups of `local` work-items; `timed`: with the event of the run.
///
/// # Safety
///
/// `queue` and `kernel` are live, and every argument of `kernel` is set to memory that outlives the run.
pub(crate) unsafe fn launch(
    queue: cl_command_queue,
    kernel: cl_kernel,
    groups: [usize; 3],
    local: [usize; 3],
    timed: bool,
) -> Result<Option<Event>, ClError> {
    let global = [
        groups[0] * local[0],
        groups[1] * local[1],
        groups[2] * local[2],
    ];
    let mut event = ptr::null_mut();
    status(
        api().clEnqueueNDRangeKernel(
            queue,
            kernel,
            3,
            ptr::null(),
            global.as_ptr(),
            local.as_ptr(),
            0,
            ptr::null(),
            if timed { &mut event } else { ptr::null_mut() },
        ),
        "clEnqueueNDRangeKernel",
    )?;
    Ok(timed.then(|| Event(event)))
}

/// An event completing with everything enqueued on `queue` so far.
///
/// # Safety
///
/// `queue` is live.
pub(crate) unsafe fn marker(queue: cl_command_queue) -> Result<Event, ClError> {
    // SAFETY: as the function's; no events to wait for.
    unsafe { cl3::command_queue::enqueue_marker_with_wait_list(queue, 0, ptr::null()) }
        .map(Event)
        .map_err(ClError::of("clEnqueueMarkerWithWaitList"))
}

/// Makes `queue`'s later commands wait for `event`.
///
/// # Safety
///
/// `queue` is live.
pub(crate) unsafe fn wait_on(queue: cl_command_queue, event: &Event) -> Result<(), ClError> {
    status(
        api().clEnqueueBarrierWithWaitList(queue, 1 as cl_uint, &event.0, ptr::null_mut()),
        "clEnqueueBarrierWithWaitList",
    )
}

/// Orders `queue`'s later commands after its earlier ones (an out-of-order queue kept in order).
///
/// # Safety
///
/// `queue` is live.
pub(crate) unsafe fn barrier(queue: cl_command_queue) -> Result<(), ClError> {
    status(
        api().clEnqueueBarrierWithWaitList(queue, 0, ptr::null(), ptr::null_mut()),
        "clEnqueueBarrierWithWaitList",
    )
}

/// Submits what `queue` holds.
pub(crate) fn flush(queue: cl_command_queue) -> Result<(), ClError> {
    cl3::command_queue::flush(queue).map_err(ClError::of("clFlush"))
}

/// Waits for everything enqueued on `queue`.
pub(crate) fn finish(queue: cl_command_queue) -> Result<(), ClError> {
    cl3::command_queue::finish(queue).map_err(ClError::of("clFinish"))
}

/// Sets argument `index` of `kernel` to the bytes of `value` (a buffer handle, a struct by value).
///
/// # Safety
///
/// `kernel` is live, argument `index` takes exactly `size_of_val(value)` bytes, and a buffer handle
/// is live and in the kernel's context.
pub(crate) unsafe fn set_arg<T: ?Sized>(
    kernel: cl_kernel,
    index: u32,
    value: &T,
) -> Result<(), ClError> {
    status(
        api().clSetKernelArg(
            kernel,
            index,
            size_of_val(value),
            (value as *const T).cast(),
        ),
        "clSetKernelArg",
    )
}

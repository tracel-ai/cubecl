//! RCCL, loaded at run time rather than linked, so a build needs nothing installed and a
//! machine without RCCL runs with no device transport.
//!
//! RCCL keeps NCCL's ABI: the declarations here are NCCL's, checked against `rccl.h`.

use core::ffi::{CStr, c_char, c_int, c_void};
use cubecl_hip_sys::hipStream_t;
use cubecl_server::command::Elements;
use libloading::Library;
use std::{
    env,
    path::{Path, PathBuf},
    sync::OnceLock,
};

/// One rank's membership of a group.
pub(crate) type Comm = *mut c_void;

/// The identifier every rank of a group joins under.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UniqueId {
    internal: [c_char; 128],
}

/// `ncclDataType_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// `ncclRedOp_t`, the reductions cubecl names.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RedOp {
    Sum = 0,
    Avg = 4,
}

/// RCCL's refusal of one call.
#[derive(Debug, Clone)]
pub(crate) struct RcclError {
    operation: &'static str,
    reason: String,
}

impl core::fmt::Display for RcclError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RCCL {} failed: {}", self.operation, self.reason)
    }
}

type Status = c_int;

/// RCCL, loaded once, with the entry points the device transport calls.
pub(crate) struct Rccl {
    get_version: unsafe extern "C" fn(*mut c_int) -> Status,
    get_unique_id: unsafe extern "C" fn(*mut UniqueId) -> Status,
    get_error_string: unsafe extern "C" fn(Status) -> *const c_char,
    comm_init_rank: unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> Status,
    all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        DataType,
        RedOp,
        Comm,
        hipStream_t,
    ) -> Status,
    all_gather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        DataType,
        Comm,
        hipStream_t,
    ) -> Status,
    reduce_scatter: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        DataType,
        RedOp,
        Comm,
        hipStream_t,
    ) -> Status,
    broadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        DataType,
        c_int,
        Comm,
        hipStream_t,
    ) -> Status,
    send: unsafe extern "C" fn(*const c_void, usize, DataType, c_int, Comm, hipStream_t) -> Status,
    recv: unsafe extern "C" fn(*mut c_void, usize, DataType, c_int, Comm, hipStream_t) -> Status,
    group_start: unsafe extern "C" fn() -> Status,
    group_end: unsafe extern "C" fn() -> Status,
    /// Keeps every entry point above loaded.
    _library: Library,
}

impl Rccl {
    /// The RCCL this process loaded, or why none did.
    pub(crate) fn get() -> Result<&'static Self, String> {
        static RCCL: OnceLock<Result<Rccl, String>> = OnceLock::new();
        // SAFETY: the library is ROCm's RCCL, whose initializers only set up its own state.
        RCCL.get_or_init(|| unsafe { open() })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The release loaded, as `ncclGetVersion` encodes it.
    pub(crate) fn version(&self) -> Result<c_int, RcclError> {
        let mut version = 0;
        // SAFETY: `version` is a valid out-pointer.
        self.check("ncclGetVersion", unsafe {
            (self.get_version)(&mut version)
        })?;
        Ok(version)
    }

    /// A fresh identifier for a group to join under.
    pub(crate) fn unique_id(&self) -> Result<UniqueId, RcclError> {
        let mut id = UniqueId { internal: [0; 128] };
        // SAFETY: `id` is a valid out-pointer to a whole `ncclUniqueId`.
        self.check("ncclGetUniqueId", unsafe { (self.get_unique_id)(&mut id) })?;
        Ok(id)
    }

    /// Joins the group `id` names as rank `rank` of `ranks`, on the current device. Blocks until
    /// every other rank joins `id` too.
    pub(crate) fn comm_init_rank(
        &self,
        ranks: c_int,
        id: UniqueId,
        rank: c_int,
    ) -> Result<Comm, RcclError> {
        let mut comm: Comm = core::ptr::null_mut();
        // SAFETY: `comm` is a valid out-pointer, and every other argument is passed by value.
        self.check("ncclCommInitRank", unsafe {
            (self.comm_init_rank)(&mut comm, ranks, id, rank)
        })?;
        Ok(comm)
    }

    /// # Safety
    ///
    /// `src` and `dst` are live device allocations of `elements`, `comm` a joined communicator,
    /// and `stream` a stream on its device.
    pub(crate) unsafe fn all_reduce(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        elements: Elements<DataType>,
        op: RedOp,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclAllReduce", unsafe {
            (self.all_reduce)(
                src,
                dst,
                elements.count,
                elements.data_type,
                op,
                comm,
                stream,
            )
        })
    }

    /// # Safety
    ///
    /// `src` holds `elements` and `dst` one `src` per rank; otherwise as
    /// [`all_reduce`](Self::all_reduce).
    pub(crate) unsafe fn all_gather(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        elements: Elements<DataType>,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclAllGather", unsafe {
            (self.all_gather)(src, dst, elements.count, elements.data_type, comm, stream)
        })
    }

    /// # Safety
    ///
    /// `dst` holds `elements` and `src` one `dst` per rank; otherwise as
    /// [`all_reduce`](Self::all_reduce).
    pub(crate) unsafe fn reduce_scatter(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        elements: Elements<DataType>,
        op: RedOp,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclReduceScatter", unsafe {
            (self.reduce_scatter)(
                src,
                dst,
                elements.count,
                elements.data_type,
                op,
                comm,
                stream,
            )
        })
    }

    /// # Safety
    ///
    /// As [`all_reduce`](Self::all_reduce); `src` is read on `root` only.
    pub(crate) unsafe fn broadcast(
        &self,
        src: *const c_void,
        dst: *mut c_void,
        elements: Elements<DataType>,
        root: c_int,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclBroadcast", unsafe {
            (self.broadcast)(
                src,
                dst,
                elements.count,
                elements.data_type,
                root,
                comm,
                stream,
            )
        })
    }

    /// # Safety
    ///
    /// `src` is a live device allocation of `elements`, `comm` a joined communicator with `peer`
    /// in it, and `stream` a stream on its device.
    pub(crate) unsafe fn send(
        &self,
        src: *const c_void,
        elements: Elements<DataType>,
        peer: c_int,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclSend", unsafe {
            (self.send)(src, elements.count, elements.data_type, peer, comm, stream)
        })
    }

    /// # Safety
    ///
    /// As [`send`](Self::send), with `dst` written.
    pub(crate) unsafe fn recv(
        &self,
        dst: *mut c_void,
        elements: Elements<DataType>,
        peer: c_int,
        comm: Comm,
        stream: hipStream_t,
    ) -> Result<(), RcclError> {
        // SAFETY: upheld by the caller.
        self.check("ncclRecv", unsafe {
            (self.recv)(dst, elements.count, elements.data_type, peer, comm, stream)
        })
    }

    /// Opens a group: the calls until [`group_end`](Self::group_end) are enqueued as one
    /// operation.
    pub(crate) fn group_start(&self) -> Result<(), RcclError> {
        // SAFETY: takes nothing.
        self.check("ncclGroupStart", unsafe { (self.group_start)() })
    }

    /// Closes the group [`group_start`](Self::group_start) opened.
    pub(crate) fn group_end(&self) -> Result<(), RcclError> {
        // SAFETY: takes nothing.
        self.check("ncclGroupEnd", unsafe { (self.group_end)() })
    }

    fn check(&self, operation: &'static str, status: Status) -> Result<(), RcclError> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: RCCL answers every status with a static, nul-terminated string.
        let message = unsafe { CStr::from_ptr((self.get_error_string)(status)) };
        Err(RcclError {
            operation,
            reason: format!("{} (status {status})", message.to_string_lossy()),
        })
    }
}

/// # Safety
///
/// Loading a library runs its initializers.
unsafe fn open() -> Result<Rccl, String> {
    let roots = ["ROCM_PATH", "HIP_PATH"]
        .into_iter()
        .filter_map(env::var_os)
        .map(PathBuf::from);
    let mut errors = Vec::new();
    for candidate in candidates(roots) {
        // SAFETY: upheld by the caller.
        match unsafe { Library::new(&candidate) } {
            // SAFETY: every symbol is resolved with its declared signature.
            Ok(library) => return unsafe { resolve(library) },
            Err(error) => errors.push(format!("{}: {error}", candidate.display())),
        }
    }
    Err(format!(
        "no RCCL could be loaded; install ROCm's `rccl` package or set ROCM_PATH:\n{}",
        errors.join("\n")
    ))
}

/// Where to look for RCCL: under each of `roots` first, then the loader's own search path. The
/// same places `cubecl-hip-sys` finds HIP, so RCCL never brings a second HIP runtime with it.
fn candidates(roots: impl Iterator<Item = PathBuf>) -> Vec<PathBuf> {
    const NAMES: [&str; 2] = ["librccl.so.1", "librccl.so"];
    let in_dir = |dir: &Path| NAMES.map(|name| dir.join(name));
    roots
        .flat_map(|root| [root.join("lib"), root])
        .flat_map(|dir| in_dir(&dir))
        .chain(NAMES.map(PathBuf::from))
        .collect()
}

/// # Safety
///
/// Each type annotation below must match the symbol's C declaration.
unsafe fn resolve(library: Library) -> Result<Rccl, String> {
    unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> Result<T, String> {
        // SAFETY: upheld by the caller of `resolve`.
        unsafe { library.get::<T>(name.as_bytes()) }
            .map(|symbol| *symbol)
            .map_err(|error| format!("RCCL has no `{name}`: {error}"))
    }
    // SAFETY: upheld by the caller.
    unsafe {
        Ok(Rccl {
            get_version: symbol(&library, "ncclGetVersion")?,
            get_unique_id: symbol(&library, "ncclGetUniqueId")?,
            get_error_string: symbol(&library, "ncclGetErrorString")?,
            comm_init_rank: symbol(&library, "ncclCommInitRank")?,
            all_reduce: symbol(&library, "ncclAllReduce")?,
            all_gather: symbol(&library, "ncclAllGather")?,
            reduce_scatter: symbol(&library, "ncclReduceScatter")?,
            broadcast: symbol(&library, "ncclBroadcast")?,
            send: symbol(&library, "ncclSend")?,
            recv: symbol(&library, "ncclRecv")?,
            group_start: symbol(&library, "ncclGroupStart")?,
            group_end: symbol(&library, "ncclGroupEnd")?,
            _library: library,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declarations_match_rccl_h() {
        assert_eq!(size_of::<UniqueId>(), 128);
        assert_eq!(size_of::<DataType>(), size_of::<c_int>());
        assert_eq!(DataType::Bfloat16 as c_int, 9);
        assert_eq!(RedOp::Avg as c_int, 4);
    }

    #[test]
    fn a_rocm_root_is_searched_before_the_loader_path() {
        let found = candidates([PathBuf::from("/rocm")].into_iter());
        assert_eq!(found[0], Path::new("/rocm/lib/librccl.so.1"));
        assert_eq!(found.last().unwrap(), Path::new("librccl.so"));
    }
}

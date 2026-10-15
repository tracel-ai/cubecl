//! NCCL's half of collectives: agreeing an identifier, joining, naming an
//! element type, and the operations.
//!
//! The bookkeeping around them — which groups this device has joined, what its
//! rank in one is — is the shared
//! [`Collectives`](cubecl_server::command::Collectives)'.

use core::ffi::c_int;
use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    ir::{DeviceProperties, ElemType, FloatKind, IntKind, UIntKind},
    server::{CommunicationId, ReduceOperation, ServerError},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Mutex;
use cubecl_server::command::{CollectiveDriver, Elements};
use cudarc::nccl::sys::ncclDataType_t as Nccl;
use std::mem::MaybeUninit;

use crate::compute::driver::Cuda;
use crate::compute::storage::gpu::GpuResource;

/// Global state map from [`CommunicationId`] to boxed [`cudarc::nccl::sys::ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

/// The NCCL reduction for `op`.
fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
    match op {
        ReduceOperation::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
        ReduceOperation::Mean => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
    }
}

/// NCCL 2.24.3, the first release with the FP8 data types.
const NCCL_FP8: c_int = 22403;

/// NCCL's FP8 data types need Hopper.
const NCCL_FP8_ARCH: u32 = 90;

/// The elements of `dtype` that `size` bytes hold, as NCCL names them.
///
/// # Errors
///
/// [`ServerError::Generic`] for an element type NCCL has no data type for, and for a size that
/// is not whole elements, whose last bytes NCCL would leave out.
fn nccl_elements(dtype: ElemType, size: u64) -> Result<Elements<Nccl>, ServerError> {
    let data_type = nccl_dtype(dtype).ok_or_else(|| ServerError::Generic {
        reason: format!("NCCL has no data type for {dtype:?}, so no collective over it"),
        backtrace: BackTrace::capture(),
    })?;
    let width = dtype.size() as u64;
    if !size.is_multiple_of(width) {
        return Err(ServerError::Generic {
            reason: format!("{size} bytes are not whole {dtype:?} elements"),
            backtrace: BackTrace::capture(),
        });
    }
    Ok(Elements {
        data_type,
        count: (size / width) as usize,
    })
}

/// Registers the device's types that the NCCL it loads moves as its device transport's.
pub fn register_nccl_transport(props: &mut DeviceProperties, arch_version: u32) {
    // SAFETY: the call only tries to `dlopen` each candidate name.
    if !unsafe { cudarc::nccl::sys::is_culib_present() } {
        return;
    }
    let Ok(version) = cudarc::nccl::result::get_nccl_version() else {
        return;
    };
    let target = NcclTarget {
        version,
        arch_version,
    };
    let moved: Vec<ElemType> = props
        .features
        .types
        .elem
        .keys()
        .copied()
        .filter(|dtype| target.moves(*dtype))
        .collect();
    for dtype in moved {
        props.register_device_transport(dtype);
    }
}

/// The NCCL loaded, which may predate types this crate names, and the device it runs on.
struct NcclTarget {
    version: c_int,
    arch_version: u32,
}

impl NcclTarget {
    fn moves(&self, dtype: ElemType) -> bool {
        match dtype {
            ElemType::Float(FloatKind::E4M3 | FloatKind::E5M2) => {
                self.version >= NCCL_FP8 && self.arch_version >= NCCL_FP8_ARCH
            }
            dtype => nccl_dtype(dtype).is_some(),
        }
    }
}

/// The NCCL data type for `dtype`.
fn nccl_dtype(dtype: ElemType) -> Option<Nccl> {
    Some(match dtype {
        ElemType::Float(FloatKind::E4M3) => Nccl::ncclFloat8e4m3,
        ElemType::Float(FloatKind::E5M2) => Nccl::ncclFloat8e5m2,
        ElemType::Float(FloatKind::F16) => Nccl::ncclFloat16,
        ElemType::Float(FloatKind::BF16) => Nccl::ncclBfloat16,
        ElemType::Float(FloatKind::F32) => Nccl::ncclFloat32,
        ElemType::Float(FloatKind::F64) => Nccl::ncclFloat64,
        ElemType::Int(IntKind::I8) => Nccl::ncclInt8,
        ElemType::Int(IntKind::I32) => Nccl::ncclInt32,
        ElemType::Int(IntKind::I64) => Nccl::ncclInt64,
        ElemType::UInt(UIntKind::U8) => Nccl::ncclUint8,
        ElemType::UInt(UIntKind::U32) => Nccl::ncclUint32,
        ElemType::UInt(UIntKind::U64) => Nccl::ncclUint64,
        // Spelled out rather than caught by a wildcard, so a new element type
        // has to be placed on one side or the other before this compiles.
        ElemType::Index
        | ElemType::Bool
        | ElemType::Float(
            FloatKind::E2M1
            | FloatKind::E2M1x2
            | FloatKind::E2M3
            | FloatKind::E3M2
            | FloatKind::UE8M0
            | FloatKind::Flex32
            | FloatKind::TF32,
        )
        | ElemType::Int(IntKind::I16)
        | ElemType::UInt(UIntKind::U16)
        | ElemType::Complex(_) => return None,
    })
}

/// NCCL's refusal to enqueue `operation`.
fn nccl_error(operation: &str, err: cudarc::nccl::result::NcclError) -> ServerError {
    ServerError::Generic {
        reason: format!("NCCL {operation} failed: {err:?}"),
        backtrace: BackTrace::capture(),
    }
}

impl CollectiveDriver for Cuda {
    type Communicator = cudarc::nccl::sys::ncclComm_t;
    type UniqueId = cudarc::nccl::sys::ncclUniqueId;
    type DataType = Nccl;
    type CommStream = cudarc::driver::sys::CUstream;

    fn group_id(id: &CommunicationId) -> Result<Self::UniqueId, ServerError> {
        let mut ids = UNIQUE_IDS_MAP.get_or_init(Default::default).lock();
        match ids.get(id) {
            Some(id) => Ok(*id),
            None => {
                let minted =
                    cudarc::nccl::result::get_uniqueid().map_err(|err| ServerError::Generic {
                        reason: format!("NCCL could not mint a communicator id: {err:?}"),
                        backtrace: BackTrace::capture(),
                    })?;
                ids.insert(id.clone(), minted);
                Ok(minted)
            }
        }
    }

    fn join(
        id: Self::UniqueId,
        ranks: usize,
        rank: usize,
    ) -> Result<Self::Communicator, ServerError> {
        let mut comm = MaybeUninit::uninit();
        // SAFETY: `comm` is a valid `MaybeUninit`, `id` is the identifier every
        // rank of this group joins under, and `rank` is this device's position
        // in it. A successful `comm_init_rank` is what makes `assume_init`
        // valid.
        unsafe {
            cudarc::nccl::result::comm_init_rank(comm.as_mut_ptr(), ranks as i32, id, rank as i32)
                .map_err(|err| ServerError::Generic {
                    reason: format!("NCCL comm_init_rank failed: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;
            Ok(comm.assume_init())
        }
    }

    fn elements(dtype: ElemType, size: u64) -> Result<Elements<Self::DataType>, ServerError> {
        nccl_elements(dtype, size)
    }

    fn all_reduce(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        op: ReduceOperation,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `comm` was joined
        // by `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::all_reduce(
                src.ptr as *const _,
                dst.ptr as *mut _,
                elements.count,
                elements.data_type,
                to_nccl_op(op),
                *comm,
                stream as _,
            )
            .map(|_| ())
            .map_err(|err| ServerError::Generic {
                reason: format!("NCCL all_reduce failed: {err:?}"),
                backtrace: BackTrace::capture(),
            })
        }
    }

    fn all_gather(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `dst` holding one `src` per rank,
        // `comm` was joined by `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::all_gather(
                src.ptr as *const _,
                dst.ptr as *mut _,
                elements.count,
                elements.data_type,
                *comm,
                stream as _,
            )
        }
        .map(|_| ())
        .map_err(|err| nccl_error("all_gather", err))
    }

    fn reduce_scatter(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        op: ReduceOperation,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `src` holding one `dst` per rank,
        // `comm` was joined by `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::reduce_scatter(
                src.ptr as *const _,
                dst.ptr as *mut _,
                elements.count,
                elements.data_type,
                to_nccl_op(op),
                *comm,
                stream as _,
            )
        }
        .map(|_| ())
        .map_err(|err| nccl_error("reduce_scatter", err))
    }

    fn broadcast(
        comm: &Self::Communicator,
        src: Option<&GpuResource>,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        root: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // NCCL reads the send buffer on the root only, so the other ranks name their own.
        let src = src.unwrap_or(dst);
        // SAFETY: both pointers are live device allocations of `elements`, `comm` was
        // joined by `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::broadcast(
                src.ptr as *const _,
                dst.ptr as *mut _,
                elements.count,
                elements.data_type,
                root as i32,
                *comm,
                stream as _,
            )
        }
        .map(|_| ())
        .map_err(|err| nccl_error("broadcast", err))
    }

    fn all_to_all(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        ranks: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        let chunk = src.size / ranks as u64;
        // NCCL has no all-to-all: one send and one receive per rank, itself included, grouped so
        // NCCL pairs them as one operation rather than waiting on each in turn.
        cudarc::nccl::result::group_start().map_err(|err| nccl_error("group_start", err))?;
        let queued = (0..ranks).try_for_each(|rank| {
            let offset = rank as u64 * chunk;
            // SAFETY: `src` and `dst` are live device allocations of `ranks` chunks of `elements`,
            // so both offsets stay inside them; `comm` was joined by `join` above, and
            // `stream` is the dedicated collective stream.
            unsafe {
                cudarc::nccl::result::send(
                    (src.ptr + offset) as *const _,
                    elements.count,
                    elements.data_type,
                    rank as i32,
                    *comm,
                    stream as _,
                )
                .map_err(|err| nccl_error("send", err))?;
                cudarc::nccl::result::recv(
                    (dst.ptr + offset) as *mut _,
                    elements.count,
                    elements.data_type,
                    rank as i32,
                    *comm,
                    stream as _,
                )
                .map_err(|err| nccl_error("recv", err))?;
            }
            Ok(())
        });
        // Closed whatever was queued: a group left open swallows every later NCCL call.
        let closed = cudarc::nccl::result::group_end().map_err(|err| nccl_error("group_end", err));
        queued.and(closed.map(|_| ()))
    }

    fn send(
        comm: &Self::Communicator,
        src: &GpuResource,
        elements: Elements<Self::DataType>,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: `src.ptr` is a live device allocation, `comm` was joined by
        // `join` above, and `stream` is the stream transfers run on.
        unsafe {
            cudarc::nccl::result::send(
                src.ptr as *const _,
                elements.count,
                elements.data_type,
                peer as i32,
                *comm,
                stream as _,
            )
            .map(|_| ())
            .map_err(|err| ServerError::Generic {
                reason: format!("NCCL send failed: {err:?}"),
                backtrace: BackTrace::capture(),
            })
        }
    }

    fn recv(
        comm: &Self::Communicator,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: `dst.ptr` is a live device allocation, `comm` was joined by
        // `join` above, and `stream` is the stream transfers run on.
        unsafe {
            cudarc::nccl::result::recv(
                dst.ptr as *mut _,
                elements.count,
                elements.data_type,
                peer as i32,
                *comm,
                stream as _,
            )
            .map(|_| ())
            .map_err(|err| ServerError::Generic {
                reason: format!("NCCL recv failed: {err:?}"),
                backtrace: BackTrace::capture(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nccl_moves_only_the_types_it_names() {
        let target = NcclTarget {
            version: 21800,
            arch_version: 75,
        };
        assert!(target.moves(ElemType::Float(FloatKind::F32)));
        assert!(target.moves(ElemType::UInt(UIntKind::U8)));
        assert!(!target.moves(ElemType::Bool));
        assert!(!target.moves(ElemType::Int(IntKind::I16)));
    }

    #[test]
    fn nccl_moves_fp8_from_its_release_on_hopper() {
        let fp8 = ElemType::Float(FloatKind::E4M3);
        let target = |version, arch_version| NcclTarget {
            version,
            arch_version,
        };
        assert!(target(NCCL_FP8, NCCL_FP8_ARCH).moves(fp8));
        assert!(!target(NCCL_FP8 - 1, NCCL_FP8_ARCH).moves(fp8));
        assert!(!target(NCCL_FP8, 89).moves(fp8));
    }

    #[test]
    fn nccl_refuses_a_partial_element() {
        let f32 = ElemType::Float(FloatKind::F32);
        assert_eq!(nccl_elements(f32, 12).map(|it| it.count).ok(), Some(3));
        assert!(nccl_elements(f32, 6).is_err());
    }
}

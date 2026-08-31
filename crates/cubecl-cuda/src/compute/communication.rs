//! NCCL's half of collectives: agreeing an identifier, joining, naming an
//! element type, and the three operations.
//!
//! The bookkeeping around them — which groups this device has joined, what its
//! rank in one is — is the shared
//! [`Collectives`](cubecl_runtime::command::Collectives)'.

use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    server::{CommunicationId, ReduceOperation, ServerError},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Mutex;
use cubecl_runtime::command::CollectiveDriver;
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

/// The NCCL data type for `dtype`, and how many of them `size` bytes hold.
///
/// # Errors
///
/// [`ServerError::Generic`] for an element type NCCL has no data type for.
/// Reported rather than fatal: a collective is one operation among many, and
/// refusing it is not a reason to take the process down — the caller can pick
/// another type, or another way to move the tensor.
fn nccl_dtype_count(
    dtype: ElemType,
    size: u64,
) -> Result<(cudarc::nccl::sys::ncclDataType_t, usize), ServerError> {
    use cudarc::nccl::sys::ncclDataType_t as Nccl;

    // Paired with the width of one element, so the count below is worked out
    // once rather than at every arm.
    let (nccl, width) = match dtype {
        ElemType::Float(FloatKind::E4M3) => (Nccl::ncclFloat8e4m3, 1),
        ElemType::Float(FloatKind::E5M2) => (Nccl::ncclFloat8e5m2, 1),
        ElemType::Float(FloatKind::F16) => (Nccl::ncclFloat16, 2),
        ElemType::Float(FloatKind::BF16) => (Nccl::ncclBfloat16, 2),
        ElemType::Float(FloatKind::F32) => (Nccl::ncclFloat32, 4),
        ElemType::Float(FloatKind::F64) => (Nccl::ncclFloat64, 8),
        ElemType::Int(IntKind::I8) => (Nccl::ncclInt8, 1),
        ElemType::Int(IntKind::I32) => (Nccl::ncclInt32, 4),
        ElemType::Int(IntKind::I64) => (Nccl::ncclInt64, 8),
        ElemType::UInt(UIntKind::U8) => (Nccl::ncclUint8, 1),
        ElemType::UInt(UIntKind::U32) => (Nccl::ncclUint32, 4),
        ElemType::UInt(UIntKind::U64) => (Nccl::ncclUint64, 8),
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
        | ElemType::Complex(_) => {
            return Err(ServerError::Generic {
                reason: format!("NCCL has no data type for {dtype:?}, so no collective over it"),
                backtrace: BackTrace::capture(),
            });
        }
    };

    Ok((nccl, (size / width) as usize))
}

impl CollectiveDriver for Cuda {
    type Communicator = cudarc::nccl::sys::ncclComm_t;
    type UniqueId = cudarc::nccl::sys::ncclUniqueId;
    type DataType = cudarc::nccl::sys::ncclDataType_t;
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

    fn data_type(dtype: ElemType, size: u64) -> Result<(Self::DataType, usize), ServerError> {
        nccl_dtype_count(dtype, size)
    }

    fn all_reduce(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        dtype: Self::DataType,
        count: usize,
        op: ReduceOperation,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `comm` was joined
        // by `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::all_reduce(
                src.ptr as *const _,
                dst.ptr as *mut _,
                count,
                dtype,
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

    fn send(
        comm: &Self::Communicator,
        src: &GpuResource,
        dtype: Self::DataType,
        count: usize,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: `src.ptr` is a live device allocation, `comm` was joined by
        // `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::send(
                src.ptr as *const _,
                count,
                dtype,
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
        dtype: Self::DataType,
        count: usize,
        peer: usize,
        stream: Self::CommStream,
    ) -> Result<(), ServerError> {
        // SAFETY: `dst.ptr` is a live device allocation, `comm` was joined by
        // `join` above, and `stream` is the dedicated collective stream.
        unsafe {
            cudarc::nccl::result::recv(
                dst.ptr as *mut _,
                count,
                dtype,
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

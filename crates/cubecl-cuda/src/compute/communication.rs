//! What NCCL needs to be told about a collective, and the identifiers that
//! keep every rank in one.

use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    device::DeviceId,
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    server::{CommunicationId, ReduceOperation, ServerError},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Mutex;

/// Global state map from [`CommunicationId`] to boxed [`cudarc::nccl::sys::ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

/// The identifier every rank of the group over `device_ids` joins under,
/// minted once and remembered.
///
/// # Errors
///
/// NCCL's refusal to mint one, which stops the group forming at all.
pub(crate) fn nccl_comm_id(
    device_ids: Vec<DeviceId>,
) -> Result<cudarc::nccl::sys::ncclUniqueId, ServerError> {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock();
    let comm_id = CommunicationId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => Ok(*id),
        None => {
            let id = cudarc::nccl::result::get_uniqueid().map_err(|err| ServerError::Generic {
                reason: format!("NCCL could not mint a communicator id: {err:?}"),
                backtrace: BackTrace::capture(),
            })?;
            unique_ids_map.insert(comm_id, id);
            Ok(id)
        }
    }
}

/// The NCCL reduction for `op`.
pub(crate) fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
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
pub(crate) fn nccl_dtype_count(
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
        | ElemType::UInt(UIntKind::U16) => {
            return Err(ServerError::Generic {
                reason: format!("NCCL has no data type for {dtype:?}, so no collective over it"),
                backtrace: BackTrace::capture(),
            });
        }
    };

    Ok((nccl, (size / width) as usize))
}

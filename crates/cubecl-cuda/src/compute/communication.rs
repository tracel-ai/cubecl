use std::{
    collections::HashMap,
    io::{Read, Write},
    mem::{self, MaybeUninit},
    net::{TcpListener, TcpStream},
    sync::OnceLock,
    time::{Duration, Instant},
};

use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    device::DeviceId,
    ir::ElemType,
    server::{ClusterInfo, CommunicationId, ReduceOperation, ServerError},
    stub::Mutex,
};

/// The `ncclUniqueId` wire size — NCCL defines this as a 128-byte opaque blob.
const NCCL_UNIQUE_ID_BYTES: usize = mem::size_of::<cudarc::nccl::sys::ncclUniqueId>();

/// How long a non-rank-0 process waits for rank 0's listener to come up during rendezvous.
const RENDEZVOUS_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
/// Backoff between connect attempts while waiting for rank 0.
const RENDEZVOUS_CONNECT_RETRY: Duration = Duration::from_millis(100);

/// In-process map from local [`CommunicationId`] to its [`ncclUniqueId`](cudarc::nccl::sys::ncclUniqueId).
///
/// The first server in a group generates the id and stores it here; subsequent servers fetch
/// the same value. This shared-memory rendezvous is what makes the local case work without an
/// external coordinator — distributed groups perform their own rendezvous inside `comm_init`.
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CommunicationId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

/// Fetch (or generate) the `ncclUniqueId` for a local communication group.
pub(crate) fn get_nccl_comm_id_local(devices: &[DeviceId]) -> cudarc::nccl::sys::ncclUniqueId {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock().unwrap();
    let comm_id = CommunicationId::local(devices);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => *id,
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}

/// Bootstrap an `ncclUniqueId` across the processes participating in a distributed group.
///
/// Rank 0 generates the id, binds to [`ClusterInfo::rendezvous_addr`], and sends the 128-byte
/// blob to each of the `world_size - 1` peers that connect. Every other rank connects to the
/// same address (with retry) and reads the 128 bytes.
///
/// This is intentionally minimal — no long-lived store, no key/value semantics: a single group
/// per endpoint, one-shot. Spin up a separate endpoint per group if you need multiple in flight.
pub(crate) fn rendezvous_distributed_unique_id(
    cluster: &ClusterInfo,
    rank: i32,
) -> Result<cudarc::nccl::sys::ncclUniqueId, ServerError> {
    if rank == 0 {
        let id = cudarc::nccl::result::get_uniqueid().map_err(|e| ServerError::Generic {
            reason: format!("NCCL get_uniqueid failed: {e:?}"),
            backtrace: BackTrace::capture(),
        })?;
        publish_unique_id(cluster, &id)?;
        Ok(id)
    } else {
        fetch_unique_id(cluster)
    }
}

fn publish_unique_id(
    cluster: &ClusterInfo,
    id: &cudarc::nccl::sys::ncclUniqueId,
) -> Result<(), ServerError> {
    let listener = TcpListener::bind(cluster.rendezvous_addr).map_err(|e| ServerError::Generic {
        reason: format!(
            "Rendezvous: rank 0 failed to bind {}: {e}",
            cluster.rendezvous_addr
        ),
        backtrace: BackTrace::capture(),
    })?;

    let bytes = unique_id_as_bytes(id);
    let peers = cluster.world_size.saturating_sub(1);
    for _ in 0..peers {
        let (mut stream, _peer) = listener.accept().map_err(|e| ServerError::Generic {
            reason: format!("Rendezvous: rank 0 accept failed: {e}"),
            backtrace: BackTrace::capture(),
        })?;
        stream
            .write_all(bytes)
            .map_err(|e| ServerError::Generic {
                reason: format!("Rendezvous: rank 0 send failed: {e}"),
                backtrace: BackTrace::capture(),
            })?;
    }
    Ok(())
}

fn fetch_unique_id(cluster: &ClusterInfo) -> Result<cudarc::nccl::sys::ncclUniqueId, ServerError> {
    let deadline = Instant::now() + RENDEZVOUS_CONNECT_TIMEOUT;
    let mut stream = loop {
        match TcpStream::connect(cluster.rendezvous_addr) {
            Ok(s) => break s,
            Err(e) if Instant::now() < deadline => {
                // Rank 0 may not be listening yet; back off and retry.
                let _ = e;
                std::thread::sleep(RENDEZVOUS_CONNECT_RETRY);
            }
            Err(e) => {
                return Err(ServerError::Generic {
                    reason: format!(
                        "Rendezvous: timed out connecting to {}: {e}",
                        cluster.rendezvous_addr
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
        }
    };

    let mut buf = [0u8; NCCL_UNIQUE_ID_BYTES];
    stream
        .read_exact(&mut buf)
        .map_err(|e| ServerError::Generic {
            reason: format!("Rendezvous: failed reading ncclUniqueId: {e}"),
            backtrace: BackTrace::capture(),
        })?;
    Ok(unique_id_from_bytes(&buf))
}

fn unique_id_as_bytes(id: &cudarc::nccl::sys::ncclUniqueId) -> &[u8] {
    // SAFETY: `ncclUniqueId` is a `#[repr(C)]` POD struct holding exactly
    // NCCL_UNIQUE_ID_BYTES of opaque bytes; reading it as a byte slice is well-defined.
    unsafe { core::slice::from_raw_parts(id as *const _ as *const u8, NCCL_UNIQUE_ID_BYTES) }
}

fn unique_id_from_bytes(bytes: &[u8; NCCL_UNIQUE_ID_BYTES]) -> cudarc::nccl::sys::ncclUniqueId {
    // SAFETY: `ncclUniqueId` is a `#[repr(C)]` POD struct of exactly NCCL_UNIQUE_ID_BYTES.
    // Writing a same-sized byte buffer into an uninit slot of that type is well-defined.
    let mut out = MaybeUninit::<cudarc::nccl::sys::ncclUniqueId>::uninit();
    unsafe {
        core::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            NCCL_UNIQUE_ID_BYTES,
        );
        out.assume_init()
    }
}

pub(crate) fn to_nccl_op(op: ReduceOperation) -> cudarc::nccl::sys::ncclRedOp_t {
    match op {
        ReduceOperation::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
        ReduceOperation::Mean => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
    }
}

pub(crate) fn get_nccl_dtype_count(
    dtype: ElemType,
    size: u64,
) -> (cudarc::nccl::sys::ncclDataType_t, usize) {
    match dtype {
        ElemType::Float(
            cubecl_core::ir::FloatKind::E2M1
            | cubecl_core::ir::FloatKind::E2M3
            | cubecl_core::ir::FloatKind::E3M2
            | cubecl_core::ir::FloatKind::UE8M0,
        ) => panic!("Minifloat not supported in NCCL"),
        ElemType::Float(cubecl_core::ir::FloatKind::E4M3) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e4m3,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::E5M2) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat8e5m2,
            size as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::F16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::BF16) => (
            cudarc::nccl::sys::ncclDataType_t::ncclBfloat16,
            (size / 2) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::Flex32) => {
            panic!("NCCL doesn't support Flex32 format.")
        }

        ElemType::Float(cubecl_core::ir::FloatKind::F32) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            (size / 4) as usize,
        ),
        ElemType::Float(cubecl_core::ir::FloatKind::TF32) => {
            panic!("NCCL doesn't support TF32 format.")
        }
        ElemType::Float(cubecl_core::ir::FloatKind::F64) => (
            cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            (size / 8) as usize,
        ),
        ElemType::Int(int_kind) => match int_kind {
            cubecl_core::ir::IntKind::I8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclInt8, size as usize)
            }
            cubecl_core::ir::IntKind::I16 => panic!("NCCL doesn't support Int16 format."),
            cubecl_core::ir::IntKind::I32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::IntKind::I64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclInt64,
                (size / 8) as usize,
            ),
        },
        ElemType::UInt(uint_kind) => match uint_kind {
            cubecl_core::ir::UIntKind::U8 => {
                (cudarc::nccl::sys::ncclDataType_t::ncclUint8, size as usize)
            }
            cubecl_core::ir::UIntKind::U16 => panic!("NCCL doesn't support UInt16 format."),
            cubecl_core::ir::UIntKind::U32 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint32,
                (size / 4) as usize,
            ),
            cubecl_core::ir::UIntKind::U64 => (
                cudarc::nccl::sys::ncclDataType_t::ncclUint64,
                (size / 8) as usize,
            ),
        },
        ElemType::Bool => panic!("NCCL doesn't support Bool format."),
    }
}

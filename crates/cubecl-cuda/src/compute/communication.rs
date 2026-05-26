use std::{
    collections::HashMap,
    io::{ErrorKind, Read, Write},
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

/// Overall deadline for the rendezvous (connect + handshake + payload) on both sides.
/// Matches PyTorch's TCPStore default and is generous enough for slow multi-host startup.
const RENDEZVOUS_DEADLINE: Duration = Duration::from_secs(300);
/// Per-stream read/write timeout — bounds how long a stalled peer can block the runner thread.
const RENDEZVOUS_IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Polling interval for the fetcher's connect retries and the publisher's accept loop.
const RENDEZVOUS_POLL_INTERVAL: Duration = Duration::from_millis(100);

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
    // Nonblocking so the accept loop can enforce an overall deadline. Each accepted stream is
    // switched back to blocking + per-IO timeout below.
    listener
        .set_nonblocking(true)
        .map_err(|e| rendezvous_err(format!("set_nonblocking on listener failed: {e}")))?;

    let bytes = unique_id_as_bytes(id);
    let expected_handshake = cluster.group_id.to_le_bytes();
    let deadline = Instant::now() + RENDEZVOUS_DEADLINE;
    let mut remaining = cluster.world_size.saturating_sub(1);

    while remaining > 0 {
        match listener.accept() {
            Ok((stream, peer)) => {
                if let Err(reason) = handshake_and_send(stream, &expected_handshake, bytes) {
                    // Wrong group / IO error on this peer — drop it and keep accepting. A
                    // misrouted connection must not consume a slot meant for a real peer.
                    tracing_skip(&peer.to_string(), &reason);
                    continue;
                }
                remaining -= 1;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(rendezvous_err(format!(
                        "rank 0 timed out waiting for peers ({remaining} of {} still missing)",
                        cluster.world_size.saturating_sub(1)
                    )));
                }
                std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
            }
            Err(e) => {
                return Err(rendezvous_err(format!("rank 0 accept failed: {e}")));
            }
        }
    }
    Ok(())
}

fn handshake_and_send(
    mut stream: TcpStream,
    expected_handshake: &[u8; 8],
    payload: &[u8],
) -> Result<(), String> {
    stream
        .set_nonblocking(false)
        .map_err(|e| format!("set_nonblocking(false): {e}"))?;
    stream
        .set_read_timeout(Some(RENDEZVOUS_IO_TIMEOUT))
        .map_err(|e| format!("set_read_timeout: {e}"))?;
    stream
        .set_write_timeout(Some(RENDEZVOUS_IO_TIMEOUT))
        .map_err(|e| format!("set_write_timeout: {e}"))?;

    let mut handshake = [0u8; 8];
    stream
        .read_exact(&mut handshake)
        .map_err(|e| format!("reading handshake: {e}"))?;
    if &handshake != expected_handshake {
        return Err(format!(
            "group_id mismatch (got {:#018x}, expected {:#018x})",
            u64::from_le_bytes(handshake),
            u64::from_le_bytes(*expected_handshake),
        ));
    }
    stream
        .write_all(payload)
        .map_err(|e| format!("sending ncclUniqueId: {e}"))?;
    Ok(())
}

fn fetch_unique_id(cluster: &ClusterInfo) -> Result<cudarc::nccl::sys::ncclUniqueId, ServerError> {
    let deadline = Instant::now() + RENDEZVOUS_DEADLINE;
    let mut stream = loop {
        match TcpStream::connect_timeout(&cluster.rendezvous_addr, RENDEZVOUS_IO_TIMEOUT) {
            Ok(s) => break s,
            // Only retry while rank 0 hasn't bound yet. Any other error is propagated as-is.
            Err(e)
                if e.kind() == ErrorKind::ConnectionRefused && Instant::now() < deadline =>
            {
                std::thread::sleep(RENDEZVOUS_POLL_INTERVAL);
            }
            Err(e) => {
                return Err(rendezvous_err(format!(
                    "timed out connecting to {}: {e}",
                    cluster.rendezvous_addr
                )));
            }
        }
    };
    stream
        .set_read_timeout(Some(RENDEZVOUS_IO_TIMEOUT))
        .map_err(|e| rendezvous_err(format!("set_read_timeout: {e}")))?;
    stream
        .set_write_timeout(Some(RENDEZVOUS_IO_TIMEOUT))
        .map_err(|e| rendezvous_err(format!("set_write_timeout: {e}")))?;

    stream
        .write_all(&cluster.group_id.to_le_bytes())
        .map_err(|e| rendezvous_err(format!("sending handshake: {e}")))?;

    let mut buf = [0u8; NCCL_UNIQUE_ID_BYTES];
    stream
        .read_exact(&mut buf)
        .map_err(|e| rendezvous_err(format!("reading ncclUniqueId: {e}")))?;
    Ok(unique_id_from_bytes(&buf))
}

fn rendezvous_err(reason: impl Into<String>) -> ServerError {
    ServerError::Generic {
        reason: format!("Rendezvous: {}", reason.into()),
        backtrace: BackTrace::capture(),
    }
}

fn tracing_skip(peer: &str, reason: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!(
        target: "cubecl_cuda::rendezvous",
        peer = peer,
        reason = reason,
        "dropping rendezvous connection"
    );
    #[cfg(not(feature = "tracing"))]
    {
        let _ = (peer, reason);
    }
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

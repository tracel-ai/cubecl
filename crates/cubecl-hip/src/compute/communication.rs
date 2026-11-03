//! RCCL's half of collectives: agreeing an identifier, joining, naming an element type, and the
//! operations.
//!
//! The bookkeeping around them is the shared [`Collectives`](cubecl_server::command::Collectives),
//! and the server's half the shared [`CollectiveServer`](cubecl_server::command::CollectiveServer).

use core::ffi::c_int;
use std::{collections::HashMap, sync::OnceLock};

use cubecl_core::{
    ir::{ElemType, FloatKind, IntKind, UIntKind},
    server::{CommunicationId, DeviceTransport, ReduceOperation, ServerError},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Mutex;
use cubecl_server::command::{CollectiveDriver, Elements, StreamSignal};

use crate::compute::driver::Hip;
use crate::compute::events::Fence;
use crate::compute::rccl::{Comm, DataType, Rccl, RcclError, RedOp, UniqueId};
use crate::compute::storage::gpu::GpuResource;

/// The identifier each group joins under, minted by whichever of its devices asks first.
static UNIQUE_IDS: OnceLock<Mutex<HashMap<CommunicationId, UniqueId>>> = OnceLock::new();

/// The transport RCCL gives a device that supports `types`, or `None`, with why logged, when
/// this process has no RCCL.
pub fn transport(types: impl Iterator<Item = ElemType>) -> Option<DeviceTransport> {
    let rccl = match Rccl::get() {
        Ok(rccl) => rccl,
        Err(reason) => {
            log::debug!("no RCCL, so transfers between HIP devices go through the host: {reason}");
            return None;
        }
    };
    match rccl.version() {
        Ok(version) => {
            log::debug!("RCCL {version} runs collectives and transfers between HIP devices");
            Some(DeviceTransport {
                reduces: types.filter(|dtype| rccl_dtype(*dtype).is_some()).collect(),
            })
        }
        Err(err) => {
            log::warn!(
                "RCCL loaded but gave no version, so transfers between HIP devices go through \
                 the host: {err}"
            );
            None
        }
    }
}

/// The RCCL data type for `dtype`.
fn rccl_dtype(dtype: ElemType) -> Option<DataType> {
    Some(match dtype {
        ElemType::Float(FloatKind::F16) => DataType::Float16,
        ElemType::Float(FloatKind::BF16) => DataType::Bfloat16,
        ElemType::Float(FloatKind::F32) => DataType::Float32,
        ElemType::Float(FloatKind::F64) => DataType::Float64,
        ElemType::Int(IntKind::I8) => DataType::Int8,
        ElemType::Int(IntKind::I32) => DataType::Int32,
        ElemType::Int(IntKind::I64) => DataType::Int64,
        ElemType::UInt(UIntKind::U8) => DataType::Uint8,
        ElemType::UInt(UIntKind::U32) => DataType::Uint32,
        ElemType::UInt(UIntKind::U64) => DataType::Uint64,
        // RCCL reduces fp8 as FNUZ on gfx942, not cubecl's OCP E4M3 and E5M2.
        ElemType::Float(FloatKind::E4M3 | FloatKind::E5M2) => return None,
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

fn to_rccl_op(op: ReduceOperation) -> RedOp {
    match op {
        ReduceOperation::Sum => RedOp::Sum,
        ReduceOperation::Mean => RedOp::Avg,
    }
}

fn rccl_error(err: RcclError) -> ServerError {
    ServerError::Generic {
        reason: err.to_string(),
        backtrace: BackTrace::capture(),
    }
}

/// One rank's membership of a group, with the RCCL that joined it.
#[derive(Clone, Copy)]
pub struct RcclComm {
    rccl: &'static Rccl,
    comm: Comm,
}

fn rccl() -> Result<&'static Rccl, ServerError> {
    Rccl::get().map_err(|reason| ServerError::Generic {
        reason,
        backtrace: BackTrace::capture(),
    })
}

impl CollectiveDriver for Hip {
    type Communicator = RcclComm;
    type UniqueId = UniqueId;
    type DataType = DataType;

    const LIBRARY: &'static str = "RCCL";

    fn wait_for(
        stream: StreamSignal<Self>,
        queued_on: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        Fence::new(queued_on).wait_async(stream)
    }

    fn group_id(id: &CommunicationId) -> Result<Self::UniqueId, ServerError> {
        let mut ids = UNIQUE_IDS.get_or_init(Default::default).lock();
        match ids.get(id) {
            Some(id) => Ok(*id),
            None => {
                let minted = rccl()?.unique_id().map_err(rccl_error)?;
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
        let rccl = rccl()?;
        let comm = rccl
            .comm_init_rank(ranks as c_int, id, rank as c_int)
            .map_err(rccl_error)?;
        Ok(RcclComm { rccl, comm })
    }

    fn data_type(dtype: ElemType) -> Option<Self::DataType> {
        rccl_dtype(dtype)
    }

    fn all_reduce(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        op: ReduceOperation,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations of `elements`, `comm` was joined by
        // `join` above, and `stream` is on its device.
        unsafe {
            comm.rccl.all_reduce(
                src.ptr,
                dst.ptr,
                elements,
                to_rccl_op(op),
                comm.comm,
                stream,
            )
        }
        .map_err(rccl_error)
    }

    fn all_gather(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `dst` holding one `src` per rank,
        // `comm` was joined by `join` above, and `stream` is on its device.
        unsafe {
            comm.rccl
                .all_gather(src.ptr, dst.ptr, elements, comm.comm, stream)
        }
        .map_err(rccl_error)
    }

    fn reduce_scatter(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        op: ReduceOperation,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // SAFETY: both pointers are live device allocations, `src` holding one `dst` per rank,
        // `comm` was joined by `join` above, and `stream` is on its device.
        unsafe {
            comm.rccl.reduce_scatter(
                src.ptr,
                dst.ptr,
                elements,
                to_rccl_op(op),
                comm.comm,
                stream,
            )
        }
        .map_err(rccl_error)
    }

    fn broadcast(
        comm: &Self::Communicator,
        src: Option<&GpuResource>,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        root: usize,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // RCCL reads the send buffer on the root only, so the other ranks name their own.
        let src = src.unwrap_or(dst);
        // SAFETY: both pointers are live device allocations of `elements`, `comm` was joined by
        // `join` above, and `stream` is on its device.
        unsafe {
            comm.rccl
                .broadcast(src.ptr, dst.ptr, elements, root as c_int, comm.comm, stream)
        }
        .map_err(rccl_error)
    }

    fn all_to_all(
        comm: &Self::Communicator,
        src: &GpuResource,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        ranks: usize,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        let (src_start, dst_start) = (src.ptr as u64, dst.ptr as u64);
        // Its sends would read chunks its receives are overwriting.
        if src_start < dst_start + dst.size && dst_start < src_start + src.size {
            return Err(ServerError::Generic {
                reason: "an all_to_all's source and destination overlap".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let rccl = comm.rccl;
        let chunk = (src.size / ranks as u64) as usize;
        // One send and one receive per rank, itself included, grouped so RCCL pairs them as one
        // operation.
        rccl.group_start().map_err(rccl_error)?;
        let queued = (0..ranks).try_for_each(|rank| {
            let offset = rank * chunk;
            // SAFETY: `src` and `dst` are live device allocations of `ranks` chunks of `elements`,
            // so both offsets stay inside them; `comm` was joined by `join` above, and `stream`
            // is on its device.
            unsafe {
                rccl.send(
                    src.ptr.wrapping_byte_add(offset),
                    elements,
                    rank as c_int,
                    comm.comm,
                    stream,
                )?;
                rccl.recv(
                    dst.ptr.wrapping_byte_add(offset),
                    elements,
                    rank as c_int,
                    comm.comm,
                    stream,
                )
            }
        });
        // Closed whatever was queued: a group left open swallows every later RCCL call.
        let closed = rccl.group_end();
        queued.and(closed).map_err(rccl_error)
    }

    fn send(
        comm: &Self::Communicator,
        src: &GpuResource,
        elements: Elements<Self::DataType>,
        peer: usize,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // SAFETY: `src.ptr` is a live device allocation, `comm` was joined by `join` above, and
        // `stream` is on its device.
        unsafe {
            comm.rccl
                .send(src.ptr, elements, peer as c_int, comm.comm, stream)
        }
        .map_err(rccl_error)
    }

    fn recv(
        comm: &Self::Communicator,
        dst: &GpuResource,
        elements: Elements<Self::DataType>,
        peer: usize,
        stream: StreamSignal<Self>,
    ) -> Result<(), ServerError> {
        // SAFETY: `dst.ptr` is a live device allocation, `comm` was joined by `join` above, and
        // `stream` is on its device.
        unsafe {
            comm.rccl
                .recv(dst.ptr, elements, peer as c_int, comm.comm, stream)
        }
        .map_err(rccl_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rccl_names_only_the_types_it_reduces() {
        assert!(rccl_dtype(ElemType::Float(FloatKind::F32)).is_some());
        assert!(rccl_dtype(ElemType::Float(FloatKind::BF16)).is_some());
        assert!(rccl_dtype(ElemType::UInt(UIntKind::U8)).is_some());
        assert!(rccl_dtype(ElemType::Bool).is_none());
        assert!(rccl_dtype(ElemType::Int(IntKind::I16)).is_none());
    }

    #[test]
    fn rccl_refuses_an_fp8_collective() {
        assert!(Hip::elements(ElemType::Float(FloatKind::E4M3), 4).is_err());
        assert!(Hip::elements(ElemType::Float(FloatKind::E5M2), 4).is_err());
    }
}

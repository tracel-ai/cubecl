use std::{
    collections::HashMap,
    mem::MaybeUninit,
    sync::{OnceLock, mpsc::Sender},
};

use cubecl_core::{device::DeviceId, stub::Mutex};
use cudarc::nccl::sys::{ncclCommGetAsyncError, ncclResult_t};

use crate::compute::communication::CommunicationAction;

#[derive(Debug, Hash, Eq, PartialEq)]
pub(crate) struct CudaCommId {
    pub str_id: String,
}

// TODO: Make sure that the peer ids are sorted.
impl From<Vec<DeviceId>> for CudaCommId {
    fn from(value: Vec<DeviceId>) -> Self {
        CudaCommId {
            str_id: value
                .iter()
                .map(|id| id.index_id.to_string())
                .collect::<Vec<String>>()
                .join(","),
        }
    }
}

/// Global state map from [`CudaCommId`] to boxed [`ncclUniqueId`].
static UNIQUE_IDS_MAP: OnceLock<Mutex<HashMap<CudaCommId, cudarc::nccl::sys::ncclUniqueId>>> =
    OnceLock::new();

pub(crate) fn get_nccl_comm_id(device_ids: Vec<DeviceId>) -> cudarc::nccl::sys::ncclUniqueId {
    let mut unique_ids_map = UNIQUE_IDS_MAP.get_or_init(Default::default).lock().unwrap();
    let comm_id = CudaCommId::from(device_ids);
    match unique_ids_map.get_mut(&comm_id) {
        Some(id) => id.clone(),
        None => {
            let id = cudarc::nccl::result::get_uniqueid().unwrap();
            unique_ids_map.insert(comm_id, id);
            id
        }
    }
}

pub(crate) struct AllReduceArgs {
    pub send_buffer: u64,
    pub recv_buffer: u64,
    pub count: usize,
    pub data_type: cudarc::nccl::sys::ncclDataType_t,
    pub op: cudarc::nccl::sys::ncclRedOp_t,
}
unsafe impl Send for AllReduceArgs {}
unsafe impl Sync for AllReduceArgs {}

pub(crate) struct CommunicationServer {
    comm: Option<*mut cudarc::nccl::sys::ncclComm>,
    stream: cudarc::nccl::sys::cudaStream_t,
}
unsafe impl Send for CommunicationServer {}
unsafe impl Sync for CommunicationServer {}

impl CommunicationServer {
    pub(crate) fn new(
        device_id: i32,
        all_ids: Vec<DeviceId>,
        stream: cudarc::nccl::sys::cudaStream_t,
    ) -> Self {
        let mut comm = MaybeUninit::uninit();
        let rank = all_ids
            .iter()
            .position(|id| id.index_id as i32 == device_id)
            .expect("Device's peer id should be in the list of device ids.");
        let nccl_comm_id = get_nccl_comm_id(all_ids.clone());
        println!("Unique id {:?}", nccl_comm_id);
        unsafe {
            println!("rank: {}", rank);
            println!("world_size: {}", all_ids.len());
            cudarc::nccl::result::comm_init_rank(
                comm.as_mut_ptr(),
                all_ids.len() as i32,
                nccl_comm_id,
                rank as i32,
            )
            .unwrap();
            Self {
                comm: Some(comm.assume_init()),
                stream,
            }
        }
    }

    pub(crate) fn process_message(&mut self, message: CommunicationAction) {
        match message {
            CommunicationAction::AllReduce(all_reduce_args) => self.all_reduce(all_reduce_args),
            CommunicationAction::Sync(sender) => self.is_finished(sender),
        }
    }

    fn is_finished(&self, sender: Sender<bool>) {
        unsafe {
            let mut state = ncclResult_t::ncclInProgress;
            println!("state: {}", state as i32);
            let res = ncclCommGetAsyncError(self.comm.unwrap(), &mut state as _);
            println!("state: {}", state as i32);
            sender.send(state as i32 == 0).unwrap();
        }
    }

    fn all_reduce(&mut self, args: AllReduceArgs) {
        println!("comm_stream all_reduce: {:?}", self.stream);
        unsafe {
            cudarc::nccl::result::all_reduce(
                args.send_buffer as *const _,
                args.recv_buffer as *mut _,
                args.count,
                args.data_type, // TODO: I need to know the type
                args.op,
                self.comm.unwrap(),
                self.stream as _,
            )
            .unwrap();
            // .map_err(|_| {
            //     IoError::Execution(ExecutionError::Generic {
            //         reason: "Error in all_reduce. Set environment variable NCCL_DEBUG to \"WARN\" for more details.".into(),
            //         backtrace: BackTrace::capture(),
            //     })
            // })?;
        }
        println!("all_reduce qd");
    }
}

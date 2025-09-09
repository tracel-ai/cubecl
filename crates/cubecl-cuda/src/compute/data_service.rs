use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

use cubecl_common::stub::Mutex;
use cubecl_core::server::IoError;
use cubecl_runtime::data_service::DataTransferId;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
use cudarc::driver::sys::{self};

use crate::compute::{CudaResource, sync::Fence};

static SERVICE: Mutex<Option<CudaDataServiceClient>> = Mutex::new(None);

pub(crate) struct CudaDataService {
    recv: Receiver<CudaDataTransferMsg>,
    transfers: HashMap<DataTransferId, CudaDataTransfer>,
}

/// A handle to a cuda resource with the context and stream for manupulating it
pub struct CudaDataTransferCall {
    pub context: sys::CUcontext,
    pub stream: sys::CUstream,
    pub resource: CudaResource,
}

unsafe impl Send for CudaDataTransferCall {}

#[derive(Clone)]
pub struct CudaDataServiceClient {
    sender: SyncSender<CudaDataTransferMsg>,
}

enum CudaDataTransferMsg {
    Send(DataTransferId, CudaDataTransferCall, u64, Fence),
    Recv(DataTransferId, CudaDataTransferCall, SyncSender<()>),
}

impl CudaDataServiceClient {
    pub fn send(
        &self,
        id: DataTransferId,
        handle: CudaDataTransferCall,
        num_bytes: u64,
        fence: Fence,
    ) {
        self.sender
            .send(CudaDataTransferMsg::Send(id, handle, num_bytes, fence))
            .unwrap();
    }

    pub fn recv(&self, id: DataTransferId, handle: CudaDataTransferCall) {
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(CudaDataTransferMsg::Recv(id, handle, send))
            .unwrap();
        recv.recv().unwrap();
    }
}

impl CudaDataService {
    /// Get a client for the Cuda data service
    pub fn get_client() -> CudaDataServiceClient {
        let mut service = SERVICE.lock().unwrap();
        if let None = *service {
            *service = Some(Self::start());
        }

        service.as_ref().unwrap().clone()
    }

    /// Launches the cuda data service, returning the first client
    fn start() -> CudaDataServiceClient {
        let (sender, recv) = std::sync::mpsc::sync_channel(32);

        let client = CudaDataServiceClient { sender };

        let service = Self {
            recv,
            transfers: HashMap::new(),
        };

        std::thread::spawn(move || service.run());

        client
    }

    /// Data service routine
    fn run(mut self) {
        while let Ok(msg) = self.recv.recv() {
            match msg {
                CudaDataTransferMsg::Send(id, handle, num_bytes, fence) => {
                    self.send(id, handle, num_bytes, fence);
                }
                CudaDataTransferMsg::Recv(id, handle, sender) => {
                    self.recv(id, handle, sender);
                }
            }
        }
    }

    fn send(
        &mut self,
        id: DataTransferId,
        call: CudaDataTransferCall,
        num_bytes: u64,
        fence: Fence,
    ) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferSendInfo {
            fence,
            call,
            num_bytes,
        };
        match transfer {
            Some(mut transfer) => {
                assert!(transfer.info_send.is_none(), "Can't send twice");

                transfer.info_send = Some(info);
                transfer.execute().unwrap();
            }
            None => {
                let mut transfer = CudaDataTransfer::default();
                transfer.info_send = Some(info);
                self.transfers.insert(id, transfer);
            }
        }
    }

    fn recv(&mut self, id: DataTransferId, call: CudaDataTransferCall, callback: SyncSender<()>) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferRecvInfo { call, callback };
        match transfer {
            Some(mut transfer) => {
                assert!(transfer.info_recv.is_none(), "Can't receive twice");
                transfer.info_recv = Some(info);

                transfer.execute().unwrap();
            }
            None => {
                let mut transfer = CudaDataTransfer::default();
                transfer.info_recv = Some(info);
                self.transfers.insert(id, transfer);
            }
        }
    }
}

#[derive(Default)]
struct CudaDataTransfer {
    info_send: Option<DataTransferSendInfo>,
    info_recv: Option<DataTransferRecvInfo>,
}

struct DataTransferSendInfo {
    fence: Fence,
    num_bytes: u64,
    call: CudaDataTransferCall,
}

struct DataTransferRecvInfo {
    call: CudaDataTransferCall,
    callback: SyncSender<()>,
}

impl CudaDataTransfer {
    fn execute(self) -> Result<(), IoError> {
        let info_send = self.info_send.expect("To be filled");
        let info_recv = self.info_recv.expect("To be filled");

        unsafe {
            cudarc::driver::result::ctx::set_current(info_recv.call.context).unwrap();

            info_send.fence.wait_async(info_recv.call.stream);

            let result = sys::cuMemcpyPeerAsync(
                info_recv.call.resource.ptr,
                info_recv.call.context,
                info_send.call.resource.ptr,
                info_send.call.context,
                info_send.num_bytes as usize,
                info_recv.call.stream,
            )
            .result();

            if let Err(_err) = result {
                // Try to enable (idempotent). If not supported, fall back.
                enable_one_way_peer_access(info_send.call.context)
                    .expect("Can't enable peer access");

                sys::cuMemcpyPeerAsync(
                    info_recv.call.resource.ptr,
                    info_recv.call.context,
                    info_send.call.resource.ptr,
                    info_send.call.context,
                    info_send.num_bytes as usize,
                    info_recv.call.stream,
                )
                .result()
                .expect("Peer communication is not activated for the provided GPUs");
            }
        };

        info_recv.callback.send(()).unwrap();

        Ok(())
    }
}

unsafe fn enable_one_way_peer_access(ctx_src: sys::CUcontext) -> Result<(), sys::CUresult> {
    unsafe {
        // Enable: destination must be current; it enables access to source.
        match sys::cuCtxEnablePeerAccess(ctx_src, 0) {
            CUDA_SUCCESS => Ok(()),
            // Already enabled → fine.
            r if r == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
            // Any other error bubbles up.
            err => Err(err),
        }
    }
}

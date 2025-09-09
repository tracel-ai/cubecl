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
    pub device: i32,
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

        // Copy from receiving device
        unsafe {
            cudarc::driver::result::ctx::set_current(info_recv.call.context).unwrap();

            // // Try to enable (idempotent). If not supported, fall back.
            // let peer_enabled = match enable_one_way_peer_access(
            //     info_recv.call.device,
            //     info_send.call.context,
            //     info_send.call.device,
            // ) {
            //     Ok(true) => true,
            //     Ok(false) => false,
            //     Err(e) if e == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => true,
            //     Err(_) => false, // conservative fallback if enable failed
            // };

            // if !peer_enabled {
            //     panic!("P2P not enabled");
            // }

            info_send.fence.wait_async(info_recv.call.stream);

            sys::cuMemcpyPeerAsync(
                info_recv.call.resource.ptr,
                info_recv.call.context,
                info_send.call.resource.ptr,
                info_send.call.context,
                info_send.num_bytes as usize,
                info_recv.call.stream,
            )
            .result()
            .unwrap();
        };

        info_recv.callback.send(()).unwrap();

        Ok(())
    }
}

fn cu_ok(res: sys::CUresult) -> Result<(), sys::CUresult> {
    if res == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(res)
    }
}

/// Enable **one-way** peer access so that memory allocated in `ctx_src`
/// is accessible from `ctx_dst`. Idempotent: returns Ok(true) if enabled
/// (either just now or previously), Ok(false) if not supported.
unsafe fn enable_one_way_peer_access(
    dev_dst: sys::CUdevice,
    ctx_src: sys::CUcontext,
    dev_src: sys::CUdevice,
) -> Result<bool, sys::CUresult> {
    unsafe {
        if !can_access_peer(dev_dst, dev_src)? {
            return Ok(false);
        }
        // Enable: destination must be current; it enables access to source.
        let res = sys::cuCtxEnablePeerAccess(ctx_src, 0);
        match res {
            CUDA_SUCCESS => Ok(true),
            // Already enabled → fine.
            r if r == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(true),
            // Any other error bubbles up.
            err => Err(err),
        }
    }
}

/// Returns true if hardware/driver supports peer access from `dev_dst` to `dev_src`.
unsafe fn can_access_peer(
    dev_dst: sys::CUdevice,
    dev_src: sys::CUdevice,
) -> Result<bool, sys::CUresult> {
    let mut can: i32 = 0;
    unsafe {
        cu_ok(sys::cuDeviceCanAccessPeer(
            &mut can as *mut i32,
            dev_dst,
            dev_src,
        ))?;
    }
    Ok(can != 0)
}

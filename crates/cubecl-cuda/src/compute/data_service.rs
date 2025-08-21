use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

use cubecl_common::stub::Mutex;
use cubecl_core::server::IoError;
use cubecl_runtime::data_service::ComputeDataTransferId;
use cudarc::driver::sys;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;

use crate::compute::{CudaResource, sync::CrossFence};

static SERVICE: Mutex<Option<CudaDataServiceClient>> = Mutex::new(None);

pub(crate) struct CudaDataService {
    recv: Receiver<CudaDataTransferMsg>,
    transfers: HashMap<ComputeDataTransferId, CudaDataTransfer>,
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
    Send(ComputeDataTransferId, CudaDataTransferCall, usize, SyncSender<()>),
    Recv(ComputeDataTransferId, CudaDataTransferCall, usize, SyncSender<()>),
}

impl CudaDataServiceClient {
    pub fn send(&self, id: ComputeDataTransferId, handle: CudaDataTransferCall, num_bytes: usize) {
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(CudaDataTransferMsg::Send(id, handle, num_bytes, send))
            .unwrap();
        recv.recv().unwrap();
    }

    pub fn recv(&self, id: ComputeDataTransferId, handle: CudaDataTransferCall, num_bytes: usize) {
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(CudaDataTransferMsg::Recv(id, handle, num_bytes, send))
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
                CudaDataTransferMsg::Send(id, handle, num_bytes, sender) => {
                    self.send(id, handle, num_bytes, sender);
                }
                CudaDataTransferMsg::Recv(id, handle, num_bytes, sender) => {
                    self.recv(id, handle, num_bytes, sender);
                }
            }
        }
    }

    fn send(&mut self, id: ComputeDataTransferId, handle: CudaDataTransferCall, num_bytes: usize, sender: SyncSender<()>) {
        let transfer = self.transfers.remove(&id);
        match transfer {
            Some(mut transfer) => {
                if transfer.src_call.is_some() {
                    panic!("Can't send twice")
                }
                transfer.src_call = Some(handle);
                transfer.src_callback = Some(sender);
                transfer.num_bytes = num_bytes;
                if transfer.dst_call.is_some() {
                    // operation is ready
                    transfer.execute().unwrap();
                } else {
                    self.transfers.insert(id, transfer);
                }
            }
            None => {
                let transfer = CudaDataTransfer {
                    src_call: Some(handle),
                    src_callback: Some(sender),
                    dst_call: None,
                    dst_callback: None,
                    num_bytes: num_bytes,
                };
                self.transfers.insert(id, transfer);
            }
        }
    }

    fn recv(&mut self, id: ComputeDataTransferId, handle: CudaDataTransferCall, num_bytes: usize, sender: SyncSender<()>) {
        let transfer = self.transfers.remove(&id);
        match transfer {
            Some(mut transfer) => {
                if transfer.dst_call.is_some() {
                    panic!("Can't receive twice")
                }
                transfer.dst_call = Some(handle);
                transfer.dst_callback = Some(sender);
                transfer.num_bytes = num_bytes;
                if transfer.src_call.is_some() {
                    // operation is ready
                    transfer.execute().unwrap();
                }
            }
            None => {
                let transfer = CudaDataTransfer {
                    src_call: None,
                    src_callback: None,
                    dst_call: Some(handle),
                    dst_callback: Some(sender),
                    num_bytes: num_bytes,
                };
                self.transfers.insert(id, transfer);
            }
        }
    }
}

struct CudaDataTransfer {
    pub(crate) src_call: Option<CudaDataTransferCall>,
    pub(crate) src_callback: Option<SyncSender<()>>,
    pub(crate) dst_call: Option<CudaDataTransferCall>,
    pub(crate) dst_callback: Option<SyncSender<()>>,
    pub(crate) num_bytes: usize,
}

impl CudaDataTransfer {
    fn execute(self) -> Result<(), IoError> {
        let CudaDataTransfer {
            src_call,
            src_callback,
            dst_call,
            dst_callback,
            num_bytes,
        } = self;

        let Some(CudaDataTransferCall {
            context: src_context,
            stream: src_stream,
            resource: src_resource,
        }) = src_call
        else {
            panic!("No source");
        };

        let Some(CudaDataTransferCall {
            context: dst_context,
            stream: dst_stream,
            resource: dst_resource,
        }) = dst_call
        else {
            panic!("No source");
        };

        // TODO could be cached
        let (dst_dev, src_dev) = unsafe {
            cudarc::driver::result::ctx::set_current(dst_context).unwrap();
            let mut dst_dev = 0;
            sys::cuCtxGetDevice(&mut dst_dev as *mut i32);
            cudarc::driver::result::ctx::set_current(src_context).unwrap();
            let mut src_dev = 0;
            sys::cuCtxGetDevice(&mut src_dev as *mut i32);

            (dst_dev, src_dev)
        };

        // Copy from receiving device, then create an event
        let transfer_fence = unsafe {
            cudarc::driver::result::ctx::set_current(dst_context).unwrap();

            // Try to enable (idempotent). If not supported, fall back.
            let peer_enabled =
                match enable_one_way_peer_access(dst_dev, src_context, src_dev) {
                    Ok(true) => true,
                    Ok(false) => false,
                    Err(e) if e == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => true,
                    Err(_) => false, // conservative fallback if enable failed
                };

            if !peer_enabled {
                panic!("P2P not enabled");
            }

            eprintln!("dst: {:?}, src: {:?}", dst_resource.ptr, src_resource.ptr);
            sys::cuMemcpyPeerAsync(
                dst_resource.ptr,
                dst_context,
                src_resource.ptr,
                src_context,
                num_bytes,
                dst_stream,
            ).result().unwrap();

            // Signal the transfer finished for the sending thread
            CrossFence::new(dst_stream, src_stream)
        };

        unsafe {
            cudarc::driver::result::ctx::set_current(src_context).unwrap();

            transfer_fence.wait();
        }

        // Operation done
        src_callback.unwrap().send(()).unwrap();
        dst_callback.unwrap().send(()).unwrap();

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

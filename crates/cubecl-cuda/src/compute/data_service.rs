use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

use cubecl_common::stub::Mutex;
use cubecl_core::server::IoError;
use cubecl_runtime::transfer::ComputeDataTransferId;
use cudarc::driver::sys;

use crate::compute::{CudaResource, sync::CrossFence};

static SERVICE: Mutex<Option<CudaDataServiceClient>> = Mutex::new(None);

pub(crate) struct CudaDataService {
    recv: Receiver<CudaDataTransferMsg>,
    transfers: HashMap<ComputeDataTransferId, CudaDataTransfer>,
}

/// A handle to a cuda resource with the context and stream for manupulating it
pub struct CudaDataHandle {
    pub context: sys::CUcontext,
    pub stream: sys::CUstream,
    pub resource: CudaResource,
}

unsafe impl Send for CudaDataHandle {}

#[derive(Clone)]
pub struct CudaDataServiceClient {
    sender: SyncSender<CudaDataTransferMsg>,
}

enum CudaDataTransferMsg {
    Send(ComputeDataTransferId, CudaDataHandle, usize),
    Recv(ComputeDataTransferId, CudaDataHandle, usize),
}

impl CudaDataServiceClient {
    pub fn send(&self, id: ComputeDataTransferId, handle: CudaDataHandle, num_bytes: usize) {
        self.sender
            .send(CudaDataTransferMsg::Send(id, handle, num_bytes))
            .unwrap();
    }

    pub fn recv(&self, id: ComputeDataTransferId, handle: CudaDataHandle, num_bytes: usize) {
        self.sender
            .send(CudaDataTransferMsg::Recv(id, handle, num_bytes))
            .unwrap();
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
                CudaDataTransferMsg::Send(id, handle, num_bytes) => {
                    self.send(id, handle, num_bytes);
                }
                CudaDataTransferMsg::Recv(id, handle, num_bytes) => {
                    self.recv(id, handle, num_bytes);
                }
            }
        }
    }

    fn send(&mut self, id: ComputeDataTransferId, handle: CudaDataHandle, num_bytes: usize) {
        let transfer = self.transfers.remove(&id);
        match transfer {
            Some(mut transfer) => {
                if transfer.src.is_some() {
                    panic!("Can't send twice")
                }
                transfer.src = Some(handle);
                transfer.num_bytes = num_bytes;
                if transfer.dst.is_some() {
                    // operation is ready
                    transfer.execute().unwrap();
                } else {
                    self.transfers.insert(id, transfer);
                }
            }
            None => {
                let transfer = CudaDataTransfer {
                    src: Some(handle),
                    dst: None,
                    num_bytes: num_bytes,
                };
                self.transfers.insert(id, transfer);
            }
        }
    }

    fn recv(&mut self, id: ComputeDataTransferId, handle: CudaDataHandle, num_bytes: usize) {
        let transfer = self.transfers.remove(&id);
        match transfer {
            Some(mut transfer) => {
                if transfer.dst.is_some() {
                    panic!("Can't receive twice")
                }
                transfer.dst = Some(handle);
                transfer.num_bytes = num_bytes;
                if transfer.src.is_some() {
                    // operation is ready
                    transfer.execute().unwrap();
                }
            }
            None => {
                let transfer = CudaDataTransfer {
                    src: None,
                    dst: Some(handle),
                    num_bytes: num_bytes,
                };
                self.transfers.insert(id, transfer);
            }
        }
    }
}

struct CudaDataTransfer {
    pub(crate) src: Option<CudaDataHandle>,
    pub(crate) dst: Option<CudaDataHandle>,
    pub(crate) num_bytes: usize,
}

impl CudaDataTransfer {
    fn execute(self) -> Result<(), IoError> {
        let CudaDataTransfer {
            src,
            dst,
            num_bytes,
        } = self;

        let Some(CudaDataHandle {
            context: src_context,
            stream: src_stream,
            resource: src_resource,
        }) = src
        else {
            panic!("No source");
        };

        let Some(CudaDataHandle {
            context: dst_context,
            stream: dst_stream,
            resource: dst_resource,
        }) = dst
        else {
            panic!("No source");
        };

        // Copy from receiving device, then create an event
        let transfer_fence = unsafe {
            cudarc::driver::result::ctx::set_current(dst_context).unwrap();

            cudarc::driver::result::memcpy_dtod_async(
                dst_resource.ptr,
                src_resource.ptr,
                num_bytes,
                dst_stream,
            )
            .unwrap();

            // Signal the transfer finished for the sending thread
            CrossFence::new(dst_stream, src_stream)
        };

        unsafe {
            cudarc::driver::result::ctx::set_current(src_context).unwrap();

            transfer_fence.wait();
        }

        Ok(())
    }
}

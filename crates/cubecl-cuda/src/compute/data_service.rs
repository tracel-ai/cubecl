use crate::compute::{storage::gpu::CudaResource, sync::Fence};
use cubecl_common::stub::Mutex;
use cubecl_core::server::IoError;
use cubecl_runtime::data_service::DataTransferId;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
use cudarc::driver::sys::{self};
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

static SERVICE: Mutex<Option<DataServiceClient>> = Mutex::new(None);

pub(crate) struct DataTransferRuntime {
    recv: Receiver<DataTransferMsg>,
    transfers: HashMap<DataTransferId, DataTransferInfo>,
}

/// A handle to a cuda resource with the context and stream.
pub struct DataTransferItem {
    pub context: sys::CUcontext,
    pub stream: sys::CUstream,
    pub resource: CudaResource,
}

unsafe impl Send for DataTransferItem {}

#[derive(Clone)]
/// The client to communicate with the async data transfer service.
pub struct DataServiceClient {
    sender: SyncSender<DataTransferMsg>,
}

/// The message of a transaction.
enum DataTransferMsg {
    Src {
        /// The unique identifier of the transaction.
        id: DataTransferId,
        /// The item to send.
        item: DataTransferItem,
        /// The fence to wait before executing the data transfer.
        fence: Fence,
    },
    Dest(DataTransferId, DataTransferItem, SyncSender<()>),
}

impl DataServiceClient {
    /// Register the source for the current data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for the current data transfer.
    /// * `item` - Source data to transfer.
    /// * `fence` - The fence to wait before executing the data transfer on the destination stream.
    ///
    /// # Panics
    ///
    /// If the data service channel is closed.
    pub fn register_src(&self, id: DataTransferId, item: DataTransferItem, fence: Fence) {
        self.sender
            .send(DataTransferMsg::Src { id, item, fence })
            .unwrap();
    }

    /// Register the destination for the current data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier for the current data transfer.
    /// * `item` - Destination to receive data.
    ///
    /// # Panics
    ///
    /// * If the data service channel is closed.
    /// * If the transfer fails.
    pub fn register_dest(&self, id: DataTransferId, item: DataTransferItem) {
        let (send, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(DataTransferMsg::Dest(id, item, send))
            .unwrap();
        recv.recv().expect("Data transfer to succeed");
    }
}

impl DataTransferRuntime {
    /// Get a client for the Cuda data service
    pub fn client() -> DataServiceClient {
        let mut service = SERVICE.lock().unwrap();
        if service.is_none() {
            *service = Some(Self::start());
        }

        service.as_ref().unwrap().clone()
    }

    fn start() -> DataServiceClient {
        let (sender, recv) = std::sync::mpsc::sync_channel(32);

        let client = DataServiceClient { sender };

        let service = Self {
            recv,
            transfers: HashMap::new(),
        };

        std::thread::spawn(move || service.run());

        client
    }

    fn run(mut self) {
        while let Ok(msg) = self.recv.recv() {
            match msg {
                DataTransferMsg::Src { id, item, fence } => {
                    self.register_src(id, item, fence);
                }
                DataTransferMsg::Dest(id, handle, sender) => {
                    self.recv(id, handle, sender);
                }
            }
        }
    }

    fn register_src(&mut self, id: DataTransferId, item: DataTransferItem, fence: Fence) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferInfoSrc { fence, item };

        match transfer {
            Some(mut transfer) => {
                assert!(transfer.info_src.is_none(), "Can't send twice");

                transfer.info_src = Some(info);
                transfer.execute().unwrap();
            }
            None => {
                let transfer = DataTransferInfo {
                    info_src: Some(info),
                    ..Default::default()
                };
                self.transfers.insert(id, transfer);
            }
        }
    }

    fn recv(&mut self, id: DataTransferId, call: DataTransferItem, callback: SyncSender<()>) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferInfoDest {
            item: call,
            callback,
        };

        match transfer {
            Some(mut transfer) => {
                assert!(transfer.info_dest.is_none(), "Can't receive twice");
                transfer.info_dest = Some(info);

                transfer.execute().unwrap();
            }
            None => {
                let transfer = DataTransferInfo {
                    info_dest: Some(info),
                    ..Default::default()
                };
                self.transfers.insert(id, transfer);
            }
        }
    }
}

#[derive(Default)]
struct DataTransferInfo {
    info_src: Option<DataTransferInfoSrc>,
    info_dest: Option<DataTransferInfoDest>,
}

struct DataTransferInfoSrc {
    fence: Fence,
    item: DataTransferItem,
}

struct DataTransferInfoDest {
    item: DataTransferItem,
    callback: SyncSender<()>,
}

impl DataTransferInfo {
    fn execute(self) -> Result<(), IoError> {
        let info_src = self.info_src.expect("To be filled");
        let info_dest = self.info_dest.expect("To be filled");

        unsafe {
            cudarc::driver::result::ctx::set_current(info_dest.item.context).unwrap();

            info_src.fence.wait_async(info_dest.item.stream);
            let num_bytes = info_dest.item.resource.size() as usize;

            let result = sys::cuMemcpyPeerAsync(
                info_dest.item.resource.ptr,
                info_dest.item.context,
                info_src.item.resource.ptr,
                info_src.item.context,
                num_bytes,
                info_dest.item.stream,
            )
            .result();

            // TODO: We should have a fallback when peer access isn't supported.
            if let Err(_err) = result {
                enable_one_way_peer_access(info_src.item.context)
                    .expect("Can't enable peer access");

                sys::cuMemcpyPeerAsync(
                    info_dest.item.resource.ptr,
                    info_dest.item.context,
                    info_src.item.resource.ptr,
                    info_src.item.context,
                    num_bytes,
                    info_dest.item.stream,
                )
                .result()
                .expect("Peer communication is not activated for the provided GPUs");
            }
        };

        info_dest.callback.send(()).unwrap();

        Ok(())
    }
}

fn enable_one_way_peer_access(ctx_src: sys::CUcontext) -> Result<(), sys::CUresult> {
    unsafe {
        match sys::cuCtxEnablePeerAccess(ctx_src, 0) {
            CUDA_SUCCESS | CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
            err => Err(err),
        }
    }
}

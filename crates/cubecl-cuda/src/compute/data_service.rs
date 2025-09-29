use crate::compute::command::{write_to_cpu, write_to_gpu};
use crate::compute::{storage::gpu::GpuResource, sync::Fence};
use cubecl_common::bytes::Bytes;
use cubecl_common::stub::Mutex;
use cubecl_core::server::{Binding, IoError};
use cubecl_runtime::data_service::DataTransferId;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
use cudarc::driver::sys::{self};
use std::{
    collections::HashMap,
    sync::mpsc::{Receiver, SyncSender},
};

static SERVICE: Mutex<Option<DataServiceClient>> = Mutex::new(None);

/// Manages asynchronous CUDA data transfers between devices.
pub(crate) struct DataTransferRuntime {
    recv: Receiver<DataTransferMsg>,
    transfers: HashMap<DataTransferId, DataTransferInfo>,
}

/// A handle to a cuda resource with the context and stream.
pub struct DataTransferItem {
    pub context: sys::CUcontext,
    pub stream: sys::CUstream,
    pub resource: GpuResource,
}

unsafe impl Send for DataTransferItem {}

#[derive(Clone)]
/// The client to communicate with the async data transfer service.
pub struct DataServiceClient {
    sender: SyncSender<DataTransferMsg>,
}

/// Message type for data transfer operations.
///
/// Supports two transfer strategies:
/// - `Peer`: Direct peer-to-peer device-to-device transfer.
/// - `Serialized`: Transfers data through CPU pinned memory.
enum DataTransferMsg {
    SrcPeer {
        id: DataTransferId,
        item: DataTransferItem,
        fence: Fence,
    },
    SrcSerialized {
        id: DataTransferId,
        item: DataTransferItem,
        /// Binding representing the original data.
        ///
        /// This needs to stay alive long enough so the read is executed on the src stream.
        binding: Binding,
    },
    DestPeer {
        id: DataTransferId,
        item: DataTransferItem,
        callback: SyncSender<()>,
    },
    DestSerialized {
        id: DataTransferId,
        item: DataTransferItem,
        bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        callback: SyncSender<()>,
    },
}

impl DataServiceClient {
    /// Registers the source for a peer-to-peer data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the data transfer.
    /// * `item` - Source data to transfer.
    /// * `fence` - Fence to synchronize before executing the transfer on the destination stream.
    ///
    /// # Panics
    ///
    /// Panics if the data service channel is closed.
    pub fn register_src_peer(&self, id: DataTransferId, item: DataTransferItem, fence: Fence) {
        self.sender
            .send(DataTransferMsg::SrcPeer { id, item, fence })
            .unwrap();
    }

    /// Registers the source for a serialized data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the data transfer.
    /// * `item` - Source data to transfer.
    /// * `binding` - Binding representing the original data, which must remain valid until the read operation completes.
    ///
    /// # Panics
    ///
    /// Panics if the data service channel is closed.
    pub fn register_src_serialized(
        &self,
        id: DataTransferId,
        item: DataTransferItem,
        binding: Binding,
    ) {
        self.sender
            .send(DataTransferMsg::SrcSerialized { id, item, binding })
            .unwrap();
    }

    /// Registers the destination for a peer-to-peer data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the data transfer.
    /// * `item` - Destination to receive the data.
    ///
    /// # Panics
    ///
    /// Panics if the data service channel is closed or if the transfer fails.
    pub fn register_dest_peer(&self, id: DataTransferId, item: DataTransferItem) {
        let (callback, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(DataTransferMsg::DestPeer { id, item, callback })
            .unwrap();
        recv.recv().expect("Data transfer to succeed");
    }

    /// Registers the destination for a serialized data transfer.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the data transfer.
    /// * `item` - Destination to receive the data.
    /// * `bytes` - Buffer containing the serialized data.
    /// * `shape` - Shape of the data tensor.
    /// * `strides` - Strides of the data tensor.
    /// * `elem_size` - Size of each element in bytes.
    ///
    /// # Panics
    ///
    /// Panics if the data service channel is closed or if the transfer fails.
    pub fn register_dest_serialized(
        &self,
        id: DataTransferId,
        item: DataTransferItem,
        bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    ) {
        let (callback, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(DataTransferMsg::DestSerialized {
                id,
                item,
                bytes,
                shape,
                strides,
                elem_size,
                callback,
            })
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
                DataTransferMsg::SrcPeer { id, item, fence } => {
                    self.register_src(id, DataTransferInfoSrc::Peer { item, fence });
                }
                DataTransferMsg::SrcSerialized { id, item, binding } => {
                    self.register_src(id, DataTransferInfoSrc::Serialized { item, binding });
                }
                DataTransferMsg::DestPeer { id, item, callback } => {
                    self.register_dest(id, DataTransferInfoDest::Peer { item, callback })
                }
                DataTransferMsg::DestSerialized {
                    id,
                    item,
                    bytes,
                    shape,
                    strides,
                    elem_size,
                    callback,
                } => self.register_dest(
                    id,
                    DataTransferInfoDest::Serialized {
                        item,
                        bytes,
                        shape,
                        strides,
                        elem_size,
                        callback,
                    },
                ),
            }
        }
    }

    fn register_src(&mut self, id: DataTransferId, info: DataTransferInfoSrc) {
        let transfer = self.transfers.remove(&id);

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

    fn register_dest(&mut self, id: DataTransferId, info: DataTransferInfoDest) {
        let transfer = self.transfers.remove(&id);

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

enum DataTransferInfoSrc {
    Serialized {
        item: DataTransferItem,
        binding: Binding,
    },
    Peer {
        item: DataTransferItem,
        fence: Fence,
    },
}

enum DataTransferInfoDest {
    Peer {
        item: DataTransferItem,
        callback: SyncSender<()>,
    },
    Serialized {
        item: DataTransferItem,
        bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        callback: SyncSender<()>,
    },
}

impl DataTransferInfo {
    fn execute(self) -> Result<(), IoError> {
        let info_src = self.info_src.expect("To be filled");
        let info_dest = self.info_dest.expect("To be filled");

        match (info_src, info_dest) {
            (
                DataTransferInfoSrc::Peer {
                    item: data_transfer_item,
                    fence,
                },
                DataTransferInfoDest::Peer { item, callback },
            ) => Self::execute_peer(fence, data_transfer_item, item, callback),
            (
                DataTransferInfoSrc::Serialized {
                    item: item_src,
                    binding,
                },
                DataTransferInfoDest::Serialized {
                    item: item_dest,
                    callback: callback_dest,
                    shape,
                    strides,
                    elem_size,
                    bytes,
                },
            ) => Self::execute_serialized(
                item_src,
                item_dest,
                bytes,
                shape,
                strides,
                elem_size,
                binding,
                callback_dest,
            ),
            _ => panic!("Invalid transfer combination"),
        }
    }

    fn execute_peer(
        src_fence: Fence,
        src_data: DataTransferItem,
        item: DataTransferItem,
        callback: SyncSender<()>,
    ) -> Result<(), IoError> {
        unsafe {
            cudarc::driver::result::ctx::set_current(item.context).unwrap();

            src_fence.wait_async(item.stream);
            let num_bytes = item.resource.size as usize;

            let result = sys::cuMemcpyPeerAsync(
                item.resource.ptr,
                item.context,
                src_data.resource.ptr,
                src_data.context,
                num_bytes,
                item.stream,
            )
            .result();

            // TODO: We should have a fallback when peer access isn't supported.
            if let Err(_err) = result {
                enable_one_way_peer_access(src_data.context).expect("Can't enable peer access");

                sys::cuMemcpyPeerAsync(
                    item.resource.ptr,
                    item.context,
                    src_data.resource.ptr,
                    src_data.context,
                    num_bytes,
                    item.stream,
                )
                .result()
                .expect("Peer communication is not activated for the provided GPUs");
            }
        };

        callback.send(()).unwrap();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_serialized(
        item_src: DataTransferItem,
        item_dest: DataTransferItem,
        mut bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        binding: Binding,
        callback_dest: SyncSender<()>,
    ) -> Result<(), IoError> {
        unsafe {
            cudarc::driver::result::ctx::set_current(item_src.context).unwrap();

            write_to_cpu(
                &shape,
                &strides,
                elem_size,
                &mut bytes,
                item_src.resource.ptr,
                item_src.stream,
            )?;
            // We can release the src binding, since the read is registered on the src stream.
            //
            // Meaning the binding could be used again on the src stream for subsequent operations.
            core::mem::drop(binding);

            let fence = Fence::new(item_src.stream);

            cudarc::driver::result::ctx::set_current(item_dest.context).unwrap();

            fence.wait_async(item_dest.stream);

            write_to_gpu(
                &shape,
                &strides,
                elem_size,
                &bytes,
                item_dest.resource.ptr,
                item_dest.stream,
            )?;
        };

        callback_dest.send(()).unwrap();

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

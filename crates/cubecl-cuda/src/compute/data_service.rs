use crate::compute::command::{write_to_cpu, write_to_gpu};
use crate::compute::{storage::gpu::GpuResource, sync::Fence};
use cubecl_common::bytes::Bytes;
use cubecl_common::stub::Mutex;
use cubecl_core::future::{self, DynFut};
use cubecl_core::server::{Binding, IoError};
use cubecl_runtime::data_service::DataTransferId;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;
use cudarc::driver::sys::{self};
use std::sync::mpsc::sync_channel;
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
    pub resource: GpuResource,
}

unsafe impl Send for DataTransferItem {}

#[derive(Clone)]
/// The client to communicate with the async data transfer service.
pub struct DataServiceClient {
    sender: SyncSender<DataTransferMsg>,
}

/// The message of a transaction.
enum DataTransferMsg {
    SrcPeer {
        /// The unique identifier of the transaction.
        id: DataTransferId,
        /// The item to send.
        item: DataTransferItem,
        /// The fence to wait before executing the data transfer.
        fence: Fence,
    },
    SrcAsync {
        /// The unique identifier of the transaction.
        id: DataTransferId,
        /// The item to send.
        item: DataTransferItem,
        /// Just to keep the original memory alive.
        binding: Binding,
        callback: SyncSender<()>,
    },
    SrcNormal {
        id: DataTransferId,
        bytes: DynFut<Bytes>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    },
    DestPeer {
        id: DataTransferId,
        item: DataTransferItem,
        callback: SyncSender<()>,
    },
    DestAsync {
        id: DataTransferId,
        item: DataTransferItem,
        bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        callback: SyncSender<()>,
    },
    DestNormal {
        id: DataTransferId,
        item: DataTransferItem,
        callback: SyncSender<()>,
    },
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
    pub fn register_src_peer(&self, id: DataTransferId, item: DataTransferItem, fence: Fence) {
        self.sender
            .send(DataTransferMsg::SrcPeer { id, item, fence })
            .unwrap();
    }

    pub fn register_src_async(&self, id: DataTransferId, item: DataTransferItem, binding: Binding) {
        let (callback, recv) = sync_channel(1);
        self.sender
            .send(DataTransferMsg::SrcAsync {
                id,
                item,
                binding,
                callback,
            })
            .unwrap();

        // recv.recv().expect("Data transfer to succeed");
    }
    pub fn register_src_normal(
        &self,
        id: DataTransferId,
        bytes: DynFut<Bytes>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    ) {
        self.sender
            .send(DataTransferMsg::SrcNormal {
                id,
                bytes,
                shape,
                strides,
                elem_size,
            })
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
    pub fn register_dest_peer(&self, id: DataTransferId, item: DataTransferItem) {
        let (callback, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(DataTransferMsg::DestPeer { id, item, callback })
            .unwrap();
        recv.recv().expect("Data transfer to succeed");
    }
    pub fn register_dest_normal(&self, id: DataTransferId, item: DataTransferItem) {
        let (callback, recv) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send(DataTransferMsg::DestNormal { id, item, callback })
            .unwrap();
        recv.recv().expect("Data transfer to succeed");
    }
    pub fn register_dest_async(
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
            .send(DataTransferMsg::DestAsync {
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
                    self.register_src_peer(id, item, fence);
                }
                DataTransferMsg::SrcAsync {
                    id,
                    item,
                    binding,
                    callback,
                } => {
                    self.register_src_async(id, item, binding, callback);
                }
                DataTransferMsg::SrcNormal {
                    id,
                    bytes,
                    shape,
                    strides,
                    elem_size,
                } => self.register_src_normal(id, bytes, shape, strides, elem_size),
                DataTransferMsg::DestPeer { id, item, callback } => {
                    self.recv(id, DataTransferInfoDest::Peer { item, callback })
                }
                DataTransferMsg::DestAsync {
                    id,
                    item,
                    bytes,
                    shape,
                    strides,
                    elem_size,
                    callback,
                } => self.recv(
                    id,
                    DataTransferInfoDest::Async {
                        item,
                        bytes,
                        shape,
                        strides,
                        elem_size,
                        callback,
                    },
                ),
                DataTransferMsg::DestNormal { id, item, callback } => {
                    self.recv(id, DataTransferInfoDest::Normal { item, callback })
                }
            }
        }
    }

    fn register_src_normal(
        &mut self,
        id: DataTransferId,
        bytes: DynFut<Bytes>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    ) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferInfoSrc::Normal {
            bytes,
            shape,
            strides,
            elem_size,
        };

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
    fn register_src_async(
        &mut self,
        id: DataTransferId,
        item: DataTransferItem,
        binding: Binding,
        callback: SyncSender<()>,
    ) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferInfoSrc::Async {
            item,
            binding,
            callback,
        };

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
    fn register_src_peer(&mut self, id: DataTransferId, item: DataTransferItem, fence: Fence) {
        let transfer = self.transfers.remove(&id);
        let info = DataTransferInfoSrc::Peer { fence, item };

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

    fn recv(&mut self, id: DataTransferId, info: DataTransferInfoDest) {
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
    Async {
        item: DataTransferItem,
        binding: Binding,
        callback: SyncSender<()>,
    },
    Peer {
        item: DataTransferItem,
        fence: Fence,
    },
    Normal {
        bytes: DynFut<Bytes>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
    },
}

enum DataTransferInfoDest {
    Peer {
        item: DataTransferItem,
        callback: SyncSender<()>,
    },
    Async {
        item: DataTransferItem,
        bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        callback: SyncSender<()>,
    },
    Normal {
        item: DataTransferItem,
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
                DataTransferInfoSrc::Async {
                    item: item_src,
                    binding,
                    callback: callback_src,
                },
                DataTransferInfoDest::Async {
                    item: item_dest,
                    callback: callback_dest,
                    shape,
                    strides,
                    elem_size,
                    bytes,
                },
            ) => Self::execute_async(
                item_src,
                item_dest,
                bytes,
                shape,
                strides,
                elem_size,
                binding,
                callback_src,
                callback_dest,
            ),
            (
                DataTransferInfoSrc::Normal {
                    bytes,
                    shape,
                    strides,
                    elem_size,
                },
                DataTransferInfoDest::Normal { item, callback },
            ) => Self::execute_normal(bytes, shape, strides, elem_size, item, callback),
            _ => unreachable!(),
        }
    }

    fn execute_normal(
        bytes: DynFut<Bytes>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        item: DataTransferItem,
        callback: SyncSender<()>,
    ) -> Result<(), IoError> {
        let bytes = future::block_on(bytes);
        let data: &[u8] = bytes.as_ref();

        unsafe {
            cudarc::driver::result::ctx::set_current(item.context).unwrap();

            write_to_gpu(
                &shape,
                &strides,
                elem_size,
                data,
                item.resource.ptr,
                item.stream,
            );
        }

        // Unblock as soon as possible.
        callback.send(()).unwrap();

        Ok(())
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

    fn execute_async(
        item_src: DataTransferItem,
        item_dest: DataTransferItem,
        mut bytes: Bytes,
        shape: Vec<usize>,
        strides: Vec<usize>,
        elem_size: usize,
        binding: Binding,
        callback_src: SyncSender<()>,
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
            );
        };

        core::mem::drop(binding);
        callback_src.send(()).unwrap();
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

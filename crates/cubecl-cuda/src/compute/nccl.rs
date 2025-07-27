#![allow(unused)]
//! **Disclaimer**: This is not safe! Most of it is, but you should definitely know or find out how many GPUs you got and don't forget to start counting at 0. With the wrong rank you give a wrong pointer. This is subject to change in the future. 
//! NCCL Operations for CubeCL CUDA Runtime
//!
//! This module provides high-level NCCL collective operations like AllReduce,
//! Broadcast, AllGather, and point-to-point communication (Send/Recv).
//!
//! **Note**: This is currently a single-threaded async variant. In the future, there will be
//! a sequential version so worker threads can use it more efficiently. Also, some grouping
//! features will be added, but these will end up in this or another abstraction layer.
//!
//! # Usage
//! ```rust,ignore
//! use cubecl_cuda::NcclOp;
//! use cubecl_core::server::Handle;
//! use cubecl_cuda::CudaRuntime;
//!
//! async fn example() {
//!     // Initialize NCCL for all available devices
//!     let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
//!     
//!     // Get handles for each device - IMPORTANT: must be sorted by device index!
//!     let targets: Vec<Handle> = get_your_handles_sorted_by_device();
//!     assert_eq!(targets.len(), nccl_op.device_count);
//!     
//!     // Perform collective operations
//!     nccl_op.all_reduce(targets.clone(), None).await;
//!     nccl_op.broadcast(targets, 0).await;
//!     
//!     // Cleanup resources
//!     nccl_op.uninit_all().await;
//! }
//! ```

use crate::CudaDevice;
use crate::runtime::CudaRuntime;
use cubecl_core::channel::MutexComputeChannel;
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::Float;
use cubecl_core::server::Handle;
use cubecl_core::{CubeElement, Runtime};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

type Client = ComputeClient<
    <CudaRuntime as Runtime>::Server,
    MutexComputeChannel<<CudaRuntime as Runtime>::Server>,
>;
type Comm = cudarc::nccl::sys::ncclComm_t;
type Stream = Arc<cudarc::nccl::sys::cudaStream_t>;

#[derive(Debug)]
pub struct NcclPtr {
    pub ptr: *mut std::ffi::c_void,
    pub count: usize,
}

pub struct NcclDevice {
    client: Client,
    comm: Option<Comm>,
    stream: Option<Stream>,
}

impl NcclDevice {
    fn new(client: Client) -> Self {
        Self {
            client,
            comm: None,
            stream: None,
        }
    }

    fn init_comm(&mut self, rank: i32, count: i32, id: cudarc::nccl::sys::ncclUniqueId) {
        let mut comm = MaybeUninit::uninit();
        let comm = unsafe {
            cudarc::nccl::result::comm_init_rank(comm.as_mut_ptr(), count, id, rank).unwrap();
            comm.assume_init()
        };

        let ordinal = unsafe { cudarc::nccl::result::comm_cu_device(comm.clone()).unwrap() };
        let device_ptr = cudarc::driver::result::device::get(ordinal).unwrap();
        let _ = unsafe {
            let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
            cudarc::driver::result::ctx::set_current(ctx).unwrap();
            ctx
        };

        let stream = cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        )
        .unwrap();
        self.comm = Some(comm);
        self.stream = Some(Arc::new(stream as _));
    }

    fn destroy(&mut self) {
        if let Some(comm) = self.comm.take() {
            unsafe {
                cudarc::nccl::result::comm_destroy(comm).ok();
            }
        }
        self.stream = None;
    }
}

pub struct NcclOp<N: Float + CubeElement> {
    map: HashMap<usize, NcclDevice>,
    device_count: usize,
    _pd: PhantomData<N>,
}

impl<N: Float + CubeElement> NcclOp<N> {
    pub fn init<R: Runtime>() -> Self {
        let mut map = HashMap::new();
        let device_count = cudarc::driver::CudaContext::device_count().unwrap() as usize;

        for device in 0..device_count {
            let cu_device = CudaDevice { index: device };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            map.insert(device, NcclDevice::new(client));
        }

        Self {
            map,
            device_count,
            _pd: PhantomData,
        }
    }

    pub async fn add_rank(mut self, rank: usize) -> Self {
        let id = cudarc::nccl::result::get_uniqueid().unwrap();

        if let Some(device) = self.map.get_mut(&rank) {
            device.init_comm(rank as i32, self.device_count as i32, id);
        }

        self
    }

    pub async fn init_all(mut self) -> Self {
        let id = cudarc::nccl::result::get_uniqueid().unwrap();
        let count = self.device_count as i32;

        cudarc::nccl::result::group_start().unwrap();

        for rank in 0..self.device_count {
            if let Some(device) = self.map.get_mut(&rank) {
                device.init_comm(rank as i32, count, id.clone());
            }
        }

        cudarc::nccl::result::group_end().unwrap();

        self
    }

    pub async fn uninit_all(mut self) {
        for (_, device) in self.map.iter_mut() {
            device.destroy();
        }
    }

    pub async fn uninit_rank(mut self, rank: usize) -> Self {
        if let Some(device) = self.map.get_mut(&rank) {
            device.destroy();
        }
        self
    }

    pub async fn all_reduce(&self, targets: Vec<Handle>, outputs: Option<Vec<Handle>>) {
        cudarc::nccl::result::group_start().unwrap();

        for (device_id, device) in &self.map {
            if let (Some(comm), Some(stream)) = (&device.comm, &device.stream) {
                if let Some(target) = targets.get(*device_id) {
                    let nccl_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, target);
                    let output_ptr = if let Some(ref output_handles) = outputs {
                        if let Some(output_handle) = output_handles.get(*device_id) {
                            NcclPtr::handle_to_nccl_ptr::<N>(&device.client, output_handle).ptr
                        } else {
                            nccl_ptr.ptr
                        }
                    } else {
                        nccl_ptr.ptr
                    };

                    let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<N>());

                    unsafe {
                        cudarc::nccl::result::all_reduce(
                            nccl_ptr.ptr as *const std::ffi::c_void,
                            output_ptr,
                            nccl_ptr.count,
                            datatype,
                            cudarc::nccl::sys::ncclRedOp_t::ncclSum,
                            *comm,
                            **stream,
                        )
                        .unwrap();
                    }
                }
            }
        }

        cudarc::nccl::result::group_end().unwrap();
    }

    pub async fn broadcast(&self, targets: Vec<Handle>, root_rank: i32) {
        cudarc::nccl::result::group_start().unwrap();

        for (device_id, device) in &self.map {
            if let (Some(comm), Some(stream)) = (&device.comm, &device.stream) {
                if let Some(target) = targets.get(*device_id) {
                    let nccl_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, target);
                    let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<N>());

                    unsafe {
                        cudarc::nccl::result::broadcast(
                            nccl_ptr.ptr as *const std::ffi::c_void,
                            nccl_ptr.ptr,
                            nccl_ptr.count,
                            datatype,
                            root_rank,
                            *comm,
                            **stream,
                        )
                        .unwrap();
                    }
                }
            }
        }

        cudarc::nccl::result::group_end().unwrap();
    }

    pub async fn all_gather(&self, send_data: Vec<Handle>, recv_data: Vec<Handle>) {
        cudarc::nccl::result::group_start().unwrap();

        for (device_id, device) in &self.map {
            if let (Some(comm), Some(stream)) = (&device.comm, &device.stream) {
                if let (Some(send), Some(recv)) =
                    (send_data.get(*device_id), recv_data.get(*device_id))
                {
                    let send_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, send);
                    let recv_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, recv);
                    let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<N>());

                    unsafe {
                        cudarc::nccl::result::all_gather(
                            send_ptr.ptr as *const std::ffi::c_void,
                            recv_ptr.ptr,
                            send_ptr.count,
                            datatype,
                            *comm,
                            **stream,
                        )
                        .unwrap();
                    }
                }
            }
        }

        cudarc::nccl::result::group_end().unwrap();
    }

    pub async fn send(&self, data: Handle, send_rank: usize, dest_rank: usize) {
        if let Some(device) = self.map.get(&send_rank) {
            if let (Some(comm), Some(stream)) = (&device.comm, &device.stream) {
                let data_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, &data);
                let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<N>());

                unsafe {
                    cudarc::nccl::result::send(
                        data_ptr.ptr as *const std::ffi::c_void,
                        data_ptr.count,
                        datatype,
                        dest_rank as i32,
                        *comm,
                        **stream,
                    )
                    .unwrap();
                }
            }
        }
    }

    pub async fn recv(&self, data: Handle, recv_rank: usize, src_rank: usize) {
        if let Some(device) = self.map.get(&recv_rank) {
            if let (Some(comm), Some(stream)) = (&device.comm, &device.stream) {
                let data_ptr = NcclPtr::handle_to_nccl_ptr::<N>(&device.client, &data);
                let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<N>());

                unsafe {
                    cudarc::nccl::result::recv(
                        data_ptr.ptr,
                        data_ptr.count,
                        datatype,
                        src_rank as i32,
                        *comm,
                        **stream,
                    )
                    .unwrap();
                }
            }
        }
    }

    pub async fn barrier(&self) {
        for (_, device) in &self.map {
            device.client.sync().await;
        }
    }

    pub fn device_count(&self) -> usize {
        self.device_count
    }
}

impl NcclPtr {
    fn handle_to_nccl_ptr<N: CubeElement>(client: &Client, handle: &Handle) -> NcclPtr {
        let binding = handle.clone().binding();
        let re_binding = client.get_resource(binding);
        let resource = re_binding.resource();
        let ptr = resource.ptr as *mut std::ffi::c_void;
        let count = resource.size() as usize / std::mem::size_of::<N>();
        NcclPtr { ptr, count }
    }

    fn get_nccl_datatype(size: usize) -> cudarc::nccl::sys::ncclDataType_t {
        match size {
            1 => cudarc::nccl::sys::ncclDataType_t::ncclInt8,
            2 => cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            4 => cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            8 => cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            _ => cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
        }
    }
}

// Drop implementation for cleanup
impl<N: Float + CubeElement> Drop for NcclOp<N> {
    fn drop(&mut self) {
        for (_, device) in self.map.iter_mut() {
            device.destroy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;
    use crate::CudaRuntime;
    use cubecl_core::CubeElement;
    use cubecl_core::prelude::*;
    use cubecl_core::server::Handle;

    fn create_test_data(client: &Client, data: Vec<f32>) -> Handle {
        client.create(f32::as_bytes(&data))
    }

    fn read_handle_data(client: &Client, handle: &Handle) -> Vec<f32> {
        let binding = handle.clone().binding();
        let bytes = client.read_one(binding);
        f32::from_bytes(&bytes).into()
    }

    #[test]
    fn test_nccl_op_initialization() {
        let nccl_op = NcclOp::<f32>::init::<CudaRuntime>();
        assert!(
            nccl_op.device_count() > 0,
            "Should have at least one CUDA device"
        );
        assert_eq!(
            nccl_op.map.len(),
            nccl_op.device_count(),
            "Map should contain all devices"
        );
    }

    #[test]
    fn test_single_device_rank_initialization() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>();
            let rank = 0;
            let initialized_op = nccl_op.add_rank(rank).await;
            if let Some(device) = initialized_op.map.get(&rank) {
                assert!(device.comm.is_some(), "Communication should be initialized");
                assert!(device.stream.is_some(), "Stream should be initialized");
            } else {
                panic!("Device at rank {} not found", rank);
            }
            initialized_op.uninit_rank(rank).await;
        };
    }

    #[test]
    fn test_all_reduce_single_device() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            let test_data = vec![1.0, 2.0, 3.0, 4.0];
            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            let input_handle = create_test_data(&client, test_data.clone());
            let targets = vec![input_handle.clone()];
            nccl_op.all_reduce(targets, None).await;
            nccl_op.barrier().await;
            let result = read_handle_data(&client, &input_handle);
            assert_eq!(
                result, test_data,
                "Single device AllReduce should preserve data"
            );
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_broadcast_single_device() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            let test_data = vec![5.0, 10.0, 15.0, 20.0];
            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            let input_handle = create_test_data(&client, test_data.clone());
            let targets = vec![input_handle.clone()];
            let root_rank = 0;
            nccl_op.broadcast(targets, root_rank).await;
            nccl_op.barrier().await;
            let result = read_handle_data(&client, &input_handle);
            assert_eq!(
                result, test_data,
                "Single device Broadcast should preserve data"
            );
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_all_gather_single_device() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            let send_data = vec![1.0, 2.0];
            let expected_recv_size = send_data.len() * nccl_op.device_count;
            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            let send_handle = create_test_data(&client, send_data.clone());
            let recv_handle = client.empty(expected_recv_size * std::mem::size_of::<f32>());
            let send_handles = vec![send_handle];
            let recv_handles = vec![recv_handle.clone()];
            nccl_op.all_gather(send_handles, recv_handles).await;
            nccl_op.barrier().await;
            let result = read_handle_data(&client, &recv_handle);
            assert_eq!(
                result[..send_data.len()],
                send_data,
                "AllGather should contain original data"
            );
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_send_recv_self() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            let test_data = vec![100.0, 200.0, 300.0];
            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            let send_handle = create_test_data(&client, test_data.clone());
            let recv_handle = client.empty(test_data.len() * std::mem::size_of::<f32>());
            let rank = 0;
            nccl_op.send(send_handle.clone(), rank, rank).await;
            nccl_op.recv(recv_handle.clone(), rank, rank).await;
            nccl_op.barrier().await;
            let result = read_handle_data(&client, &recv_handle);
            assert_eq!(result, test_data, "Send/Recv to self should preserve data");
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_nccl_ptr_functionality() {
        let cu_device = CudaDevice { index: 0 };
        let client = <CudaRuntime as Runtime>::client(&cu_device);
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let handle = create_test_data(&client, test_data.clone());
        let nccl_ptr = NcclPtr::handle_to_nccl_ptr::<f32>(&client, &handle);
        assert!(!nccl_ptr.ptr.is_null(), "Pointer should not be null");
        assert_eq!(
            nccl_ptr.count,
            test_data.len(),
            "Count should match data length"
        );
        let datatype = NcclPtr::get_nccl_datatype(std::mem::size_of::<f32>());
        assert_eq!(
            datatype,
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            "Should detect f32 type"
        );
    }

    #[test]
    fn test_barrier_functionality() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            nccl_op.barrier().await;
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_different_data_types() {
        let f16_size = 2;
        let f32_size = 4;
        let f64_size = 8;
        let i8_size = 1;
        assert_eq!(
            NcclPtr::get_nccl_datatype(f16_size),
            cudarc::nccl::sys::ncclDataType_t::ncclFloat16
        );
        assert_eq!(
            NcclPtr::get_nccl_datatype(f32_size),
            cudarc::nccl::sys::ncclDataType_t::ncclFloat32
        );
        assert_eq!(
            NcclPtr::get_nccl_datatype(f64_size),
            cudarc::nccl::sys::ncclDataType_t::ncclFloat64
        );
        assert_eq!(
            NcclPtr::get_nccl_datatype(i8_size),
            cudarc::nccl::sys::ncclDataType_t::ncclInt8
        );
    }

    #[test]
    fn test_performance_benchmark() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;

            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);

            // Test mit verschiedenen Datengrößen
            let sizes = vec![100, 1000, 10000, 100000];

            for size in sizes {
                let test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let handle = create_test_data(&client, test_data);
                let targets = vec![handle];
                nccl_op.all_reduce(targets, None).await;
                nccl_op.barrier().await;
            }
            nccl_op.uninit_all().await;
        };
    }

    #[test]
    fn test_memory_safety() {
        async {
            let nccl_op = NcclOp::<f32>::init::<CudaRuntime>().init_all().await;
            let cu_device = CudaDevice { index: 0 };
            let client = <CudaRuntime as Runtime>::client(&cu_device);
            let mut handles = Vec::new();
            for i in 0..10 {
                let data = vec![i as f32; 100];
                let handle = create_test_data(&client, data);
                handles.push(handle);
            }
            for handle in &handles {
                let targets = vec![handle.clone()];
                nccl_op.all_reduce(targets, None).await;
            }
            nccl_op.barrier().await;
            for (i, handle) in handles.iter().enumerate() {
                let result = read_handle_data(&client, handle);
                assert_eq!(result[0], i as f32, "Data should be preserved");
            }
            nccl_op.uninit_all().await;
        };
    }
}

//* Note that `cudarc::driver::result::init().unwrap()` will not longer be handled
//* in `crate::runtime::create_client` when nccl is activated.
#![allow(unused)]
use crate::CudaDevice;
use crate::compute::CudaServer;
use crate::runtime::CudaRuntime;
use cubecl_core::channel::MutexComputeChannel;
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::Float;
use cubecl_core::server::Handle;
use cubecl_core::{CubeElement, Runtime};
use cudarc::driver::result::ctx::set_current;
use cudarc::nccl::sys::ncclUniqueId;
use futures::executor::block_on;
use futures::future::join_all;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

type Client = ComputeClient<
    <CudaRuntime as Runtime>::Server,
    MutexComputeChannel<<CudaRuntime as Runtime>::Server>,
>;

type Comm = cudarc::nccl::sys::ncclComm_t;
type Stream = cudarc::driver::sys::CUstream;

/// This struct encapsulates the functionality of a single NCCL device and provides methods to perform
/// collective communication operations like all-reduce, broadcast, reduce, send, and receive.
pub struct NcclDevice {
    client: Client,
    comm: Comm,
    rank: i32,
}

impl NcclDevice {
    fn new(client: Client, comm: Comm, rank: i32) -> Self {
        Self { client, comm, rank }
    }

    /// Creates a new `NcclDevice` instance for the given rank.
    pub fn init(rank: usize, id: ncclUniqueId) -> Self {
        let client = <CudaRuntime as Runtime>::client(&CudaDevice { index: rank });
        let count = <CudaRuntime as Runtime>::device_count() as i32;
        let mut comm = MaybeUninit::uninit();
        let comm = unsafe {
            cudarc::nccl::result::comm_init_rank(comm.as_mut_ptr(), count, id, rank as i32)
                .unwrap();
            comm.assume_init()
        };
        Self {
            client,
            comm,
            rank: rank as i32,
        }
    }

    /// Ensures all ongoing operations have completed.
    pub fn barrier(&self) {
        block_on(self.client.sync());
    }

    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
    pub fn all_reduce<N: Float + CubeElement>(
        &self,
        input: &Handle,
        output: Option<&Handle>,
        op: ReduceOp,
    ) {
        let in_ptr = NcclPtr::<N>::get(self.client(), input);
        let out_ptr = output
            .map(|h| NcclPtr::<N>::get(self.client(), h))
            .unwrap_or_else(|| in_ptr);

        unsafe {
            cudarc::nccl::result::all_reduce(
                in_ptr.ptr as *const ::core::ffi::c_void,
                out_ptr.ptr,
                in_ptr.count,
                in_ptr.nccl_type(),
                op.convert(),
                self.comm,
                in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .unwrap();
        }
    }

    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast
    pub fn broadcast<N: Float + CubeElement>(
        &self,
        input: &Handle,
        output: Option<&Handle>,
        root: i32,
    ) {
        let in_ptr = NcclPtr::<N>::get(self.client(), input);
        let out_ptr = output
            .map(|h| NcclPtr::<N>::get(self.client(), h))
            .unwrap_or_else(|| in_ptr);
        unsafe {
            cudarc::nccl::result::broadcast(
                in_ptr.ptr as *const ::core::ffi::c_void,
                out_ptr.ptr,
                in_ptr.count,
                in_ptr.nccl_type(),
                root,
                self.comm,
                in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .unwrap();
        }
    }

    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reduce
    pub fn reduce<N: Float + CubeElement>(
        &self,
        input: &Handle,
        output: Option<&Handle>,
        op: ReduceOp,
        root: i32,
    ) {
        let in_ptr = NcclPtr::<N>::get(&self.client, input);
        let out_ptr = output
            .map(|h| NcclPtr::<N>::get(&self.client, h))
            .unwrap_or_else(|| in_ptr);
        unsafe {
            cudarc::nccl::result::reduce(
                in_ptr.ptr as *const ::core::ffi::c_void,
                out_ptr.ptr,
                in_ptr.count,
                in_ptr.nccl_type(),
                op.convert(),
                root,
                self.comm,
                in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .unwrap();
        }
    }

    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
    pub fn send<N: Float + CubeElement>(&self, send_buffer: &Handle, peer: i32) {
        let send_ptr = NcclPtr::<N>::get(self.client(), send_buffer);
        unsafe {
            cudarc::nccl::result::send(
                send_ptr.ptr as *const ::core::ffi::c_void,
                send_ptr.count,
                send_ptr.nccl_type(),
                peer,
                self.comm,
                send_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .unwrap();
        }
    }

    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
    pub fn recv<N: Float + CubeElement>(&self, recv_buffer: &Handle, peer: i32) {
        let recv_ptr = NcclPtr::<N>::get(self.client(), recv_buffer);
        unsafe {
            cudarc::nccl::result::recv(
                recv_ptr.ptr,
                recv_ptr.count,
                recv_ptr.nccl_type(),
                peer,
                self.comm,
                recv_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .unwrap();
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

pub struct LazyNccl {
    nccl_devs: Vec<NcclDevice>,
    device_count: usize,
}

impl LazyNccl {
    /// Creates a single group with all devices on a single thread.
    pub fn new() -> Self {
        cudarc::driver::result::init().unwrap();
        let count = <CudaRuntime as Runtime>::device_count();
        let mut clients = Vec::new();
        let mut comms = vec![std::ptr::null_mut(); count];
        let ordinals: Vec<i32> = Vec::from_iter(0i32..count as i32);
        for n in 0..count {
            clients.push(<CudaRuntime as Runtime>::client(&CudaDevice { index: n }));
        }
        unsafe {
            cudarc::nccl::result::comm_init_all(
                comms.as_mut_ptr(),
                count as i32,
                ordinals.as_ptr(),
            )
            .unwrap();
        }
        let nccls = comms
            .into_iter()
            .zip(clients.into_iter())
            .enumerate()
            .map(|(rank, (comm, client))| NcclDevice::new(client, comm, rank as i32))
            .collect();

        Self {
            nccl_devs: nccls,
            device_count: count,
        }
    }

    /// Copies the data to every device.
    pub fn create_copies(&self, data: &[u8]) -> Vec<Handle> {
        let mut handles = Vec::new();
        for client in &self.nccl_devs {
            handles.push(client.client().create(data));
        }
        handles
    }

    /// Assumes the `handles` vecotor got a `Handle` for each device in order of the rank.
    pub fn read_intervall(&self, handles: Vec<Handle>) -> Vec<Vec<u8>> {
        self.nccl_devs
            .iter()
            .zip(handles.into_iter())
            .map(|(nccl, handle)| nccl.client().read_one(handle))
            .collect()
    }

    /// Waits till all devices are synced.
    pub fn barrier(&self) {
        block_on(async {
            let sync_futures: Vec<_> = self
                .nccl_devs
                .iter()
                .map(|client| client.client().sync())
                .collect();
            let _ = join_all(sync_futures).await;
        });
    }

    /// Assumes the `handles` vecotor got a `Handle` for each device in order of the rank.
    /// If outputs is `None` reduction will be in place.
    pub fn all_reduce<N: Float + CubeElement>(
        &self,
        inputs: &[Handle],
        outputs: Option<&Vec<Handle>>,
        op: ReduceOp,
    ) {
        cudarc::nccl::result::group_start().unwrap();
        for nccl_client in &self.nccl_devs {
            let rank = nccl_client.rank as usize;
            let in_ptr = NcclPtr::<N>::get(nccl_client.client(), &inputs[rank]);
            let out_ptr = outputs
                .and_then(|o| o.get(rank))
                .map(|h| NcclPtr::<N>::get(nccl_client.client(), h))
                .unwrap_or_else(|| in_ptr);
            unsafe {
                cudarc::nccl::result::all_reduce(
                    in_ptr.ptr as *const ::core::ffi::c_void,
                    out_ptr.ptr,
                    in_ptr.count,
                    in_ptr.nccl_type(),
                    op.convert(),
                    nccl_client.comm,
                    in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
                )
                .unwrap();
            }
        }
        cudarc::nccl::result::group_end().unwrap();
    }

    /// Assumes the `handles` vecotor got a `Handle` for each device in order of the rank.
    /// If outputs is `None` reduction will be in place.
    pub fn broadcast<N: Float + CubeElement>(
        &self,
        inputs: &[Handle],
        outputs: Option<&Vec<Handle>>,
        root: i32,
    ) {
        cudarc::nccl::result::group_start().unwrap();
        for nccl_client in &self.nccl_devs {
            let rank = nccl_client.rank as usize;
            let in_ptr = NcclPtr::<N>::get(nccl_client.client(), &inputs[rank]);
            let out_ptr = outputs
                .and_then(|o| o.get(rank))
                .map(|h| NcclPtr::<N>::get(nccl_client.client(), h))
                .unwrap_or_else(|| in_ptr);
            unsafe {
                cudarc::nccl::result::broadcast(
                    in_ptr.ptr as *const ::core::ffi::c_void,
                    out_ptr.ptr,
                    in_ptr.count,
                    in_ptr.nccl_type(),
                    root,
                    nccl_client.comm,
                    in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
                )
                .unwrap();
            }
        }
        cudarc::nccl::result::group_end().unwrap();
    }

    /// Assumes the `handles` vecotor got a `Handle` for each device in order of the rank.
    /// If outputs is `None` reduction will be in place.
    pub fn reduce<N: Float + CubeElement>(
        &self,
        inputs: &[Handle],
        outputs: Option<&Vec<Handle>>,
        op: ReduceOp,
        root: i32,
    ) {
        cudarc::nccl::result::group_start().unwrap();
        for nccl_client in &self.nccl_devs {
            let rank = nccl_client.rank as usize;
            let in_ptr = NcclPtr::<N>::get(nccl_client.client(), &inputs[rank]);
            let out_ptr = outputs
                .and_then(|o| o.get(rank))
                .map(|h| NcclPtr::<N>::get(nccl_client.client(), h))
                .unwrap_or_else(|| in_ptr);
            unsafe {
                cudarc::nccl::result::reduce(
                    in_ptr.ptr as *const ::core::ffi::c_void,
                    out_ptr.ptr,
                    in_ptr.count,
                    in_ptr.nccl_type(),
                    op.convert(),
                    root,
                    nccl_client.comm,
                    in_ptr.stream as *mut cudarc::nccl::sys::CUstream_st,
                )
                .unwrap();
            }
        }
        cudarc::nccl::result::group_end().unwrap();
    }

    pub fn send_recv(&self, from: usize, get: &Handle, to: usize, put: &Handle) {
        cudarc::nccl::result::group_start().unwrap();
        self.nccl_devs[from].send::<f32>(get, to as i32);
        self.nccl_devs[to].recv::<f32>(put, from as i32);
        cudarc::nccl::result::group_end().unwrap();
    }

    pub fn get_client(&self, rank: usize) -> &Client {
        self.nccl_devs[rank].client()
    }
}

#[derive(Debug, Clone, Copy)]
struct NcclPtr<N: Float + CubeElement> {
    ptr: *mut std::ffi::c_void,
    stream: Stream,
    count: usize,
    _pd: PhantomData<N>,
}

impl<N: Float + CubeElement> NcclPtr<N> {
    fn get(client: &Client, handle: &Handle) -> Self {
        let binding = handle.clone().binding();
        let re_binding = client.get_resource(binding);
        let resource = re_binding.resource();
        let count = resource.size() as usize / size_of::<N>();
        NcclPtr {
            ptr: resource.ptr as *mut std::ffi::c_void,
            stream: resource.stream,
            count,
            _pd: PhantomData::<N>,
        }
    }

    fn nccl_type(&self) -> cudarc::nccl::sys::ncclDataType_t {
        match std::mem::size_of::<N>() {
            1 => cudarc::nccl::sys::ncclDataType_t::ncclInt8,
            2 => cudarc::nccl::sys::ncclDataType_t::ncclFloat16,
            4 => cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
            8 => cudarc::nccl::sys::ncclDataType_t::ncclFloat64,
            _ => cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
        }
    }
}

#[derive(Debug)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

impl ReduceOp {
    fn convert(&self) -> cudarc::nccl::sys::ncclRedOp_t {
        match self {
            ReduceOp::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
            ReduceOp::Prod => cudarc::nccl::sys::ncclRedOp_t::ncclProd,
            ReduceOp::Max => cudarc::nccl::sys::ncclRedOp_t::ncclMax,
            ReduceOp::Min => cudarc::nccl::sys::ncclRedOp_t::ncclMin,
            ReduceOp::Avg => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::prelude::*;
    use std::mem::size_of;

    #[test]
    fn all_reduce() {
        let group = LazyNccl::new();
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let handles = group.create_copies(f32::as_bytes(data.as_slice()));
        group.all_reduce::<f32>(&handles, None, ReduceOp::Sum);
        group.barrier();
        let results = group.read_intervall(handles);
        let expected = vec![
            group.device_count as f32 * 1.0,
            group.device_count as f32 * 2.0,
            group.device_count as f32 * 3.0,
            group.device_count as f32 * 4.0,
        ];
        for result in results {
            let values = f32::from_bytes(result.as_slice());
            assert_eq!(values, expected);
        }
    }

    #[test]
    fn broadcast() {
        let group = LazyNccl::new();
        let mut handles = Vec::new();
        for i in 0..group.device_count {
            let data = vec![i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32];
            handles.push(group.get_client(i).create(f32::as_bytes(data.as_slice())));
        }
        let root = 0;
        group.broadcast::<f32>(&handles, None, root);
        group.barrier();
        let results = group.read_intervall(handles);
        let expected = vec![0.0f32, 1.0f32, 2.0f32, 3.0f32];
        for result in results {
            let values = f32::from_bytes(result.as_slice());
            assert_eq!(values, expected);
        }
    }

    #[test]
    fn reduce() {
        let group = LazyNccl::new();
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let handles = group.create_copies(f32::as_bytes(data.as_slice()));
        let root = 0;
        group.reduce::<f32>(&handles, None, ReduceOp::Sum, root);
        group.barrier();
        let results = group.read_intervall(handles);
        let expected = vec![
            group.device_count as f32 * 1.0,
            group.device_count as f32 * 2.0,
            group.device_count as f32 * 3.0,
            group.device_count as f32 * 4.0,
        ];
        for (i, result) in results.iter().enumerate() {
            let values = f32::from_bytes(result.as_slice());
            if i == root as usize {
                assert_eq!(values, expected);
            }
        }
    }

    #[test]
    fn send_recv() {
        let group = LazyNccl::new();
        if group.device_count >= 2 {
            let send_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
            let send_handle = group
                .get_client(0)
                .create(f32::as_bytes(send_data.as_slice()));
            let recv_handle = group.get_client(1).empty(16);
            group.send_recv(0, &send_handle, 1, &recv_handle);
            group.barrier();
            let recv_result = group.get_client(1).read_one(recv_handle);
            let recv_values = f32::from_bytes(recv_result.as_slice());
            assert_eq!(recv_values, send_data);
        }
    }

    #[test]
    fn reduce_operations() {
        let group = LazyNccl::new();
        let n = group.device_count as f32;
        let data = vec![2.0f32, 3.0f32, 4.0f32, 5.0f32];
        let test_cases = vec![
            (ReduceOp::Sum, vec![n * 2.0, n * 3.0, n * 4.0, n * 5.0]),
            (
                ReduceOp::Prod,
                vec![
                    2.0f32.powi(n as i32),
                    3.0f32.powi(n as i32),
                    4.0f32.powi(n as i32),
                    5.0f32.powi(n as i32),
                ],
            ),
            (ReduceOp::Max, vec![2.0, 3.0, 4.0, 5.0]),
            (ReduceOp::Min, vec![2.0, 3.0, 4.0, 5.0]),
            (ReduceOp::Avg, vec![2.0, 3.0, 4.0, 5.0]),
        ];
        for (op, expected) in test_cases {
            let handles = group.create_copies(f32::as_bytes(data.as_slice()));
            group.all_reduce::<f32>(&handles, None, op);
            group.barrier();
            let results = group.read_intervall(handles);
            let values = f32::from_bytes(results[0].as_slice());
            assert_eq!(values, expected);
        }
    }
}

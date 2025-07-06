// Usage example:
/*
pub async fn example_usage() {
    // Get your CUDA clients
    let clients: Vec<&Client> = get_cuda_clients();

    // Create some data handles
    let handle1 = clients[0].create(&[14, 51, 111, 103]);
    let handle2 = clients[0].create(&[14, 51, 111, 103]);

    // Initialize NCCL operations
    let nccl_op = NcclOp::init(clients);

    // Execute operations first element represents a group
    nccl_op.all_reduce(0, handle1, None).await;
    nccl_op.send(0, handle1, 1).await;
    nccl_op.recv(0, handle2, 0).await;

    // Barrier synchronization
    nccl_op.barrier(0).await;

    // Cleanup when done
    nccl_op.uninit().await;
}
*/

use crate::nccl::device::get_stream;
use crate::runtime::CudaRuntime;
use cubecl_core::Runtime;
use cubecl_core::channel::MutexComputeChannel;
use cubecl_core::client::ComputeClient;
use cubecl_core::server::Handle;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::mem::transmute;

type Client = ComputeClient<
    <CudaRuntime as Runtime>::Server,
    MutexComputeChannel<<CudaRuntime as Runtime>::Server>,
>;
type Comm = cudarc::nccl::sys::ncclComm_t;

#[derive(Debug)]
pub struct NcclPtr {
    pub ptr: *mut std::ffi::c_void,
    pub size: usize,
}

pub struct NcclOp {
    client_map: HashMap<usize, Vec<(Comm, Client)>>,
    initialized: bool,
}

impl NcclOp {
    pub async fn init(clients: Vec<&Client>) -> Self {
        let mut client_map: HashMap<usize, Vec<(Comm, Client)>> = HashMap::new();

        for client in clients {
            client.sync().await;
            let device = client.info();
            let group_id = device.nccl.group();
            let count = device.nccl.count();
            let id = device.nccl.id();
            let rank = device.index as i32;

            let mut comm = MaybeUninit::uninit();
            let comm = unsafe {
                cudarc::nccl::result::comm_init_rank(
                    comm.as_mut_ptr(),
                    count.try_into().unwrap(),
                    id,
                    rank,
                )
                .unwrap();
                comm.assume_init()
            };

            client_map
                .entry(group_id)
                .or_insert_with(Vec::new)
                .push((comm, client.clone()));
        }

        Self {
            client_map,
            initialized: true,
        }
    }

    /// Uninitialize and cleanup all NCCL resources
    pub async fn uninit(mut self) {
        if !self.initialized {
            return;
        }

        for (_, group_clients) in &mut self.client_map {
            for (comm, client) in group_clients {
                unsafe {
                    cudarc::nccl::result::comm_destroy(*comm).unwrap();
                }

                client.sync().await;
            }
        }

        self.client_map.clear();
        self.initialized = false;
    }

    /// Execute AllReduce operation on a specific group with optional output handle.
    pub async fn all_reduce(&self, group_id: usize, target: Handle, output: Option<Handle>) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        for (comm, client) in group_clients {
            let device = client.info();
            let nccl_ptr = handle_to_nccl_ptr(client, &target);
            let output_ptr = if let Some(ref output_handle) = output {
                handle_to_nccl_ptr(client, output_handle).ptr
            } else {
                nccl_ptr.ptr
            };
            let stream = get_stream(&device).unwrap();
            let datatype = get_nccl_datatype(device.nccl.size());
            let count = device.nccl.count();

            unsafe {
                cudarc::nccl::result::all_reduce(
                    nccl_ptr.ptr as *const std::ffi::c_void,
                    output_ptr,
                    count,
                    datatype,
                    cudarc::nccl::sys::ncclRedOp_t::ncclSum,
                    *comm,
                    transmute(stream),
                )
                .unwrap();
            }
        }
    }

    /// Execute Broadcast operation
    pub async fn broadcast(&self, group_id: usize, target: Handle, root_rank: i32) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        for (comm, client) in group_clients {
            let device = client.info();
            let nccl_ptr = handle_to_nccl_ptr(client, &target);
            let stream = get_stream(&device).unwrap();
            let datatype = get_nccl_datatype(device.nccl.size());
            let count = device.nccl.count();

            unsafe {
                cudarc::nccl::result::broadcast(
                    nccl_ptr.ptr as *const std::ffi::c_void,
                    nccl_ptr.ptr,
                    count,
                    datatype,
                    root_rank,
                    *comm,
                    transmute(stream),
                )
                .unwrap();
            }
        }
    }

    /// Execute AllGather operation
    pub async fn all_gather(&self, group_id: usize, send_data: Handle, recv_data: Handle) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        for (comm, client) in group_clients {
            let device = client.info();
            let send_ptr = handle_to_nccl_ptr(client, &send_data);
            let recv_ptr = handle_to_nccl_ptr(client, &recv_data);
            let stream = get_stream(&device).unwrap();
            let datatype = get_nccl_datatype(device.nccl.size());
            let send_count = device.nccl.count();

            unsafe {
                cudarc::nccl::result::all_gather(
                    send_ptr.ptr as *const std::ffi::c_void,
                    recv_ptr.ptr,
                    send_count,
                    datatype,
                    *comm,
                    transmute(stream),
                )
                .unwrap();
            }
        }
    }

    /// Execute Send operation
    pub async fn send(&self, group_id: usize, data: Handle, send_rank: usize, dest_rank: usize) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        let (comm, client) = &group_clients[send_rank];
        let device = client.info();
        let data_ptr = handle_to_nccl_ptr(client, &data);
        let stream = get_stream(&device).unwrap();
        let datatype = get_nccl_datatype(device.nccl.size());
        let count = device.nccl.count();

        unsafe {
            cudarc::nccl::result::send(
                data_ptr.ptr as *const std::ffi::c_void,
                count,
                datatype,
                dest_rank.try_into().unwrap(),
                *comm,
                transmute(stream),
            )
            .unwrap();
        }
    }

    /// Execute Recv operation
    pub async fn recv(&self, group_id: usize, data: Handle, recv_rank: usize, src_rank: usize) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        let (comm, client) = &group_clients[recv_rank];
        let device = client.info();
        let data_ptr = handle_to_nccl_ptr(client, &data);
        let stream = get_stream(&device).unwrap();
        let datatype = get_nccl_datatype(device.nccl.size());
        let count = device.nccl.count();

        unsafe {
            cudarc::nccl::result::recv(
                data_ptr.ptr,
                count,
                datatype,
                src_rank.try_into().unwrap(),
                *comm,
                transmute(stream),
            )
            .unwrap();
        }
    }

    /// Execute barrier synchronization
    pub async fn barrier(&self, group_id: usize) {
        if !self.initialized {
            return;
        }

        let group_clients = self.client_map.get(&group_id).unwrap();

        for (_, client) in group_clients {
            client.sync().await;
        }
    }
}

fn handle_to_nccl_ptr(client: &Client, handle: &Handle) -> NcclPtr {
    let binding = handle.clone().binding();
    let re_binding = client.get_resource(binding);
    let resource = re_binding.resource();
    let ptr = resource.ptr as *mut std::ffi::c_void;
    let size = resource.size();

    NcclPtr {
        ptr,
        size: size as usize,
    }
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

// Drop implementation for cleanup
impl Drop for NcclOp {
    fn drop(&mut self) {
        if self.initialized {
            for (_, group_clients) in &mut self.client_map {
                for (comm, _) in group_clients {
                    unsafe {
                        let _ = cudarc::nccl::result::comm_destroy(*comm);
                    }
                }
            }
            self.client_map.clear();
            self.initialized = false;
        }
    }
}

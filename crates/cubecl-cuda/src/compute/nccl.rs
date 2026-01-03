#![allow(dead_code)]
#![allow(missing_docs)]
use std::mem::MaybeUninit;

use cubecl_core::server::Binding;
use cubecl_runtime::plugin::{Plugin, PluginError, PluginType};
use cudarc::nccl::sys::{
    ncclComm_t, ncclDataType_t, ncclRedOp_t, ncclScalarResidence_t, ncclUniqueId,
};

#[derive(Debug)]
pub struct NcclComm(*mut cudarc::nccl::sys::ncclComm);

unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

impl NcclComm {
    pub fn as_ptr(&self) -> ncclComm_t {
        self.0
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                cudarc::nccl::result::comm_destroy(self.0).ok();
            }
        }
    }
}

pub struct NcclExtension;

/// The types are defined which are used by client and server end.
impl Plugin for NcclExtension {
    type ClientHandle = NcclClientHandle;
    type ServerHandle = NcclServerHandle;
    type InitType = NcclInit;
    type ReturnVal = ();
    type Fns = fn(Self::ServerHandle) -> Result<(), PluginError>;
    const EXTENSION_NAME: &'static str = "cuda_nccl";
}

/// Info needed to initialize a NcclComm.
#[derive(Debug, Clone)]
pub struct NcclInit {
    /// Needs to be id = Device::device_count_total() - 1
    pub id: i32,
    /// Device::device_count_total()
    pub dev_count: i32,
    /// cudarc::nccl::result::get_uniqueid().unwrap()
    pub uid: ncclUniqueId,
}

/// Here the `PluginType` is used to initialize `Nccl`.
/// `NcclComm` is used as the `Insert` type and gets injected
/// into `CudaServer` when called through `ComputeClient`'s new `plugin_init()`
impl PluginType for NcclInit {
    type Insert = NcclComm;

    /// Function used to generate the Insert type
    fn init(self) -> Self::Insert {
        let mut comm = MaybeUninit::uninit();
        let comm = unsafe {
            cudarc::nccl::result::comm_init_rank(
                comm.as_mut_ptr(),
                self.dev_count,
                self.uid,
                self.id,
            )
            .unwrap();
            comm.assume_init()
        };
        NcclComm(comm)
    }
}

#[derive(new, Debug, Clone)]
pub struct NcclClientHandle {
    /// For example with broadcast is no input needed.
    /// Thus resulting in the use of `Option`.
    pub input: Option<Binding>,
    /// Also `Option` for send and receive.
    pub output: Option<Binding>,
    /// Device::device_count_total()
    pub device_count: usize,
    /// cudarc::nccl::result::get_uniqueid().unwrap()
    pub nccl_type: ncclDataType_t,
}

unsafe impl Send for NcclServerHandle {}

/// This struct will be constructed by `CudaServer`,
/// when `client.plugin_fn(c: NcclClientHandle)`
pub struct NcclServerHandle {
    pub input: Option<*const ::core::ffi::c_void>,
    pub output: Option<*mut std::ffi::c_void>,
    pub dev_count: usize,
    pub ty: ncclDataType_t,
    pub comm: ncclComm_t,
    pub stream: cudarc::driver::sys::CUstream,
}

impl NcclServerHandle {
    pub fn all_reduce(self, op: ReduceOp) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::all_reduce(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                op.convert(),
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("all_reduce: {:?}", e)))
        }
    }

    pub fn broadcast(self, root: i32) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::broadcast(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                root,
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("broadcast: {:?}", e)))
        }
    }

    pub fn reduce(self, op: ReduceOp, root: i32) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::reduce(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                op.convert(),
                root,
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("reduce: {:?}", e)))
        }
    }

    pub fn reduce_scatter(self, op: ReduceOp) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::reduce_scatter(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                op.convert(),
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("reduce_scatter: {:?}", e)))
        }
    }

    pub fn all_gather(self) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::all_gather(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("all_gather: {:?}", e)))
        }
    }

    pub fn send(self, peer: i32) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::send(
                self.input
                    .ok_or_else(|| PluginError::InvalidHandle("Input required".into()))?,
                self.dev_count,
                self.ty,
                peer,
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("send: {:?}", e)))
        }
    }

    pub fn recv(self, peer: i32) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::recv(
                self.output
                    .ok_or_else(|| PluginError::InvalidHandle("Output required".into()))?,
                self.dev_count,
                self.ty,
                peer,
                self.comm,
                self.stream as *mut cudarc::nccl::sys::CUstream_st,
            )
            .map(|_| ())
            .map_err(|e| PluginError::ExecutionFailed(format!("recv: {:?}", e)))
        }
    }

    pub fn create_custom_pre_mul_sum(
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> Result<ncclRedOp_t, PluginError> {
        unsafe {
            let mut op = MaybeUninit::uninit();
            cudarc::nccl::result::reduce_op_create_pre_mul_sum(
                op.as_mut_ptr(),
                scalar,
                datatype,
                residence,
                comm,
            )
            .map(|_| op.assume_init())
            .map_err(|e| {
                PluginError::ExecutionFailed(format!("create_custom_pre_mul_sum: {:?}", e))
            })
        }
    }

    pub fn destroy_custom_op(op: ncclRedOp_t, comm: ncclComm_t) -> Result<(), PluginError> {
        unsafe {
            cudarc::nccl::result::reduce_op_destroy(op, comm)
                .map(|_| ())
                .map_err(|e| PluginError::ExecutionFailed(format!("destroy_custom_op: {:?}", e)))
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Reduce operation type. 
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
    Custom(ncclRedOp_t),
}

impl ReduceOp {
    fn convert(&self) -> ncclRedOp_t {
        match self {
            ReduceOp::Sum => cudarc::nccl::sys::ncclRedOp_t::ncclSum,
            ReduceOp::Prod => cudarc::nccl::sys::ncclRedOp_t::ncclProd,
            ReduceOp::Max => cudarc::nccl::sys::ncclRedOp_t::ncclMax,
            ReduceOp::Min => cudarc::nccl::sys::ncclRedOp_t::ncclMin,
            ReduceOp::Avg => cudarc::nccl::sys::ncclRedOp_t::ncclAvg,
            ReduceOp::Custom(op) => *op,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ScalarResidence {
    Device,
    Host,
}

impl ScalarResidence {
    fn to_nccl(&self) -> ncclScalarResidence_t {
        match self {
            ScalarResidence::Device => cudarc::nccl::sys::ncclScalarResidence_t::ncclScalarDevice,
            ScalarResidence::Host => {
                cudarc::nccl::sys::ncclScalarResidence_t::ncclScalarHostImmediate
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::CudaServer;
    use crate::{CudaDevice, CudaRuntime};
    use cubecl_common::device::Device;
    use cubecl_core::prelude::*;
    use cubecl_runtime::client::ComputeClient;
    use std::sync::Mutex;

    static NCCL_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn has_multi_gpu() -> bool {
        let dev_count = CudaDevice::device_count_total();
        println!("{}", dev_count);
        dev_count >= 2
    }

    fn init_nccl_pair() -> Option<(
        ComputeClient<CudaServer>,
        ComputeClient<CudaServer>,
        cudarc::nccl::sys::ncclUniqueId,
    )> {
        if !has_multi_gpu() {
            println!("Skipping test: requires at least 2 GPUs");
            return None;
        }

        let device0 = CudaDevice::new(0);
        let device1 = CudaDevice::new(1);

        let client0 = CudaRuntime::client(&device0);
        let client1 = CudaRuntime::client(&device1);

        let uid = cudarc::nccl::result::get_uniqueid().unwrap();

        Some((client0, client1, uid))
    }

    fn setup_nccl_communicators(
        client0: &ComputeClient<CudaServer>,
        client1: &ComputeClient<CudaServer>,
        uid: cudarc::nccl::sys::ncclUniqueId,
    ) {
        let init0 = NcclInit {
            id: 0,
            dev_count: 2,
            uid,
        };
        let init1 = NcclInit {
            id: 1,
            dev_count: 2,
            uid,
        };

        std::thread::scope(|s| {
            let h0 = s.spawn(|| {
                client0
                    .plugin_init::<NcclExtension>(init0)
                    .expect("Failed to initialize NCCL on device 0");
            });

            let h1 = s.spawn(|| {
                client1
                    .plugin_init::<NcclExtension>(init1)
                    .expect("Failed to initialize NCCL on device 1");
            });

            h0.join().unwrap();
            h1.join().unwrap();
        });
    }

    fn run_nccl_op(
        client0: &ComputeClient<CudaServer>,
        client1: &ComputeClient<CudaServer>,
        uid: cudarc::nccl::sys::ncclUniqueId,
        handle0: NcclClientHandle,
        handle1: NcclClientHandle,
        op: fn(NcclServerHandle) -> Result<(), PluginError>,
    ) {
        let init0 = NcclInit {
            id: 0,
            dev_count: 2,
            uid,
        };
        let init1 = NcclInit {
            id: 1,
            dev_count: 2,
            uid,
        };

        std::thread::scope(|s| {
            let h0 = s.spawn(move || {
                client0
                    .plugin_init::<NcclExtension>(init0)
                    .expect("Failed to initialize NCCL on device 0");
                client0
                    .plugin_fn::<NcclExtension>(handle0, op)
                    .expect("NCCL operation on GPU 0 failed");
            });

            let h1 = s.spawn(move || {
                client1
                    .plugin_init::<NcclExtension>(init1)
                    .expect("Failed to initialize NCCL on device 1");
                client1
                    .plugin_fn::<NcclExtension>(handle1, op)
                    .expect("NCCL operation on GPU 1 failed");
            });

            h0.join().unwrap();
            h1.join().unwrap();
        });
    }

    #[test]
    fn test_nccl_all_reduce_sum_f32() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let expected: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(data0.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(data1.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            data1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let all_reduce_sum = |handle: NcclServerHandle| -> Result<(), PluginError> {
            handle.all_reduce(ReduceOp::Sum)
        };

        run_nccl_op(&client0, &client1, uid, handle0, handle1, all_reduce_sum);

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result1_bytes = client1.read(vec![output1]);

        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result0, expected, "GPU 0 result mismatch");
        assert_eq!(result1, expected, "GPU 1 result mismatch");
    }

    #[test]
    fn test_nccl_broadcast() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let data1: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(data0.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(data1.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            data1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let broadcast_root_0 =
            |handle: NcclServerHandle| -> Result<(), PluginError> { handle.broadcast(0) };

        run_nccl_op(&client0, &client1, uid, handle0, handle1, broadcast_root_0);

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result1_bytes = client1.read(vec![output1]);

        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result0, data0, "GPU 0 result mismatch");
        assert_eq!(result1, data0, "GPU 1 result mismatch");
    }

    #[test]
    fn test_nccl_reduce() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let expected: Vec<f32> = vec![6.0, 8.0, 10.0, 12.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(data0.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(data1.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            data1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let reduce_sum_root_0 = |handle: NcclServerHandle| -> Result<(), PluginError> {
            handle.reduce(ReduceOp::Sum, 0)
        };

        run_nccl_op(&client0, &client1, uid, handle0, handle1, reduce_sum_root_0);

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();

        assert_eq!(result0, expected, "GPU 0 (root) result mismatch");
    }

    #[test]
    fn test_nccl_reduce_scatter() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data1: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let expected0: Vec<f32> = vec![6.0, 8.0];
        let expected1: Vec<f32> = vec![10.0, 12.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(expected0.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(expected1.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            expected0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            expected1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let reduce_scatter_sum = |handle: NcclServerHandle| -> Result<(), PluginError> {
            handle.reduce_scatter(ReduceOp::Sum)
        };

        run_nccl_op(
            &client0,
            &client1,
            uid,
            handle0,
            handle1,
            reduce_scatter_sum,
        );

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result1_bytes = client1.read(vec![output1]);

        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result0, expected0, "GPU 0 result mismatch");
        assert_eq!(result1, expected1, "GPU 1 result mismatch");
    }

    #[test]
    fn test_nccl_all_gather() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![1.0, 2.0];
        let data1: Vec<f32> = vec![3.0, 4.0];

        let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(expected.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(expected.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            data1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let all_gather =
            |handle: NcclServerHandle| -> Result<(), PluginError> { handle.all_gather() };

        run_nccl_op(&client0, &client1, uid, handle0, handle1, all_gather);

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result1_bytes = client1.read(vec![output1]);

        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result0, expected, "GPU 0 result mismatch");
        assert_eq!(result1, expected, "GPU 1 result mismatch");
    }

    #[test]
    fn test_nccl_send_recv() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output1 = client1.empty(data0.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            None,
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            None,
            Some(output1.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let init0 = NcclInit {
            id: 0,
            dev_count: 2,
            uid,
        };
        let init1 = NcclInit {
            id: 1,
            dev_count: 2,
            uid,
        };

        std::thread::scope(|s| {
            let h0 = s.spawn(|| {
                client0
                    .plugin_init::<NcclExtension>(init0)
                    .expect("Failed to initialize NCCL on device 0");
                client0
                    .plugin_fn::<NcclExtension>(handle0, |handle: NcclServerHandle| handle.send(1))
                    .expect("Send on GPU 0 failed");
            });

            let h1 = s.spawn(|| {
                client1
                    .plugin_init::<NcclExtension>(init1)
                    .expect("Failed to initialize NCCL on device 1");
                client1
                    .plugin_fn::<NcclExtension>(handle1, |handle: NcclServerHandle| handle.recv(0))
                    .expect("Recv on GPU 1 failed");
            });

            h0.join().unwrap();
            h1.join().unwrap();
        });

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result1_bytes = client1.read(vec![output1]);
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result1, data0, "GPU 1 received data mismatch");
    }

    #[test]
    fn test_nccl_all_reduce_with_different_ops() {
        let _lock = NCCL_TEST_LOCK.lock().unwrap();
        let Some((client0, client1, uid)) = init_nccl_pair() else {
            return;
        };

        let data0: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
        let data1: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];

        let expected_max: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];

        let input0 = client0.create_from_slice(bytemuck::cast_slice(&data0));
        let output0 = client0.empty(data0.len() * std::mem::size_of::<f32>());

        let input1 = client1.create_from_slice(bytemuck::cast_slice(&data1));
        let output1 = client1.empty(data1.len() * std::mem::size_of::<f32>());

        let handle0 = NcclClientHandle::new(
            Some(input0.clone().binding()),
            Some(output0.clone().binding()),
            data0.len(),
            ncclDataType_t::ncclFloat32,
        );

        let handle1 = NcclClientHandle::new(
            Some(input1.clone().binding()),
            Some(output1.clone().binding()),
            data1.len(),
            ncclDataType_t::ncclFloat32,
        );

        let all_reduce_max = |handle: NcclServerHandle| -> Result<(), PluginError> {
            handle.all_reduce(ReduceOp::Max)
        };

        run_nccl_op(&client0, &client1, uid, handle0, handle1, all_reduce_max);

        cubecl_common::reader::read_sync(client0.sync());
        cubecl_common::reader::read_sync(client1.sync());

        let result0_bytes = client0.read(vec![output0]);
        let result1_bytes = client1.read(vec![output1]);

        let result0: Vec<f32> = bytemuck::cast_slice(&result0_bytes[0]).to_vec();
        let result1: Vec<f32> = bytemuck::cast_slice(&result1_bytes[0]).to_vec();

        assert_eq!(result0, expected_max, "GPU 0 result mismatch for Max op");
        assert_eq!(result1, expected_max, "GPU 1 result mismatch for Max op");
    }
}

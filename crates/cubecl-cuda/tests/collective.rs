use cubecl_common::{bytes::Bytes, device::Device};
use cubecl_core::{
    ir::{ElemType, FloatKind},
    server::ReduceOperation,
};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_server::runtime::Runtime;
use std::sync::{Arc, Barrier};

#[test]
#[ignore = "requires two CUDA devices"]
fn cuda_collective_operations_keep_the_same_order() {
    let device0 = CudaDevice::new(0);
    let device1 = CudaDevice::new(1);
    assert!(CudaRuntime::enumerate_devices(0).len() >= 2);

    let device_ids = vec![device0.to_id(), device1.to_id()];
    let barrier = Arc::new(Barrier::new(3));
    let dtype = ElemType::Float(FloatKind::F32);

    std::thread::scope(|scope| {
        let barrier = barrier.clone();
        let device0 = device0.clone();
        let device1 = device1.clone();
        scope.spawn(move || {
            let mut source = CudaRuntime::client(&device0);
            let mut destination = CudaRuntime::client(&device1);
            for value in 0..256 {
                barrier.wait();
                let input = [value as f32; 4];
                let source_handle = source.create_from_slice(f32::as_bytes(&input));
                let destination_handle = source.to_client(source_handle, &destination, dtype);
                let returned_handle = destination.to_client(destination_handle, &source, dtype);
                let returned = source.read_one(returned_handle).unwrap();
                assert_eq!(f32::from_bytes(&returned), input);
            }
        });

        for (rank, device) in [device0, device1].into_iter().enumerate() {
            let barrier = barrier.clone();
            let device_ids = device_ids.clone();
            scope.spawn(move || {
                let mut client = CudaRuntime::client(&device);
                for _ in 0..256 {
                    barrier.wait();
                    let handle = client.create_from_slice(f32::as_bytes(&[rank as f32; 4]));
                    client.all_reduce(
                        handle.clone(),
                        handle.clone(),
                        dtype,
                        device_ids.clone(),
                        ReduceOperation::Mean,
                    );
                    client.sync_collective();
                    let output = client.read_one(handle).unwrap();
                    assert_eq!(f32::from_bytes(&output), &[0.5; 4]);
                }
            });
        }
    });
}

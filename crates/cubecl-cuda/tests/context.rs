use cubecl_common::bytes::Bytes;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_server::runtime::Runtime;

#[test]
fn restores_current_context_after_external_changes() {
    let client = CudaRuntime::client(&CudaDevice::default());
    let expected = client
        .exclusive(|| cudarc::driver::result::ctx::get_current().unwrap().unwrap() as usize)
        .unwrap();

    for foreign in [false, true] {
        let _foreign_context = client
            .exclusive(move || {
                if foreign {
                    Some(cudarc::driver::CudaContext::new_non_primary(0, 0).unwrap())
                } else {
                    // SAFETY: a null context clears this worker's current context.
                    // The primary context remains retained by the runtime.
                    unsafe {
                        cudarc::driver::result::ctx::set_current(core::ptr::null_mut()).unwrap();
                    }
                    None
                }
            })
            .unwrap();

        for _ in 0..2 {
            let values = vec![1_u8, 7, 19, 255];
            let handle = client.create(Bytes::from_bytes_vec(values.clone()));
            let actual = client.read_one(handle).unwrap();
            assert_eq!(&actual[..], values.as_slice());
            assert_eq!(
                client
                    .exclusive(|| {
                        cudarc::driver::result::ctx::get_current().unwrap().unwrap() as usize
                    })
                    .unwrap(),
                expected,
            );
        }
    }
}

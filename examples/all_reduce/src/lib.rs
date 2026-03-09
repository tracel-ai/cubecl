mod all_reduce_ex {
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    use cubecl::stub::Mutex;
    use cudarc::driver::CudaContext;
    use cudarc::driver::sys::{CUevent_st, CUstream_st};
    use cudarc::nccl::Comm;

    pub fn hello_world() {
        println!("Hello world;");
    }

    #[derive(Clone)]
    struct StreamWrapper(pub *mut CUstream_st);
    unsafe impl Send for StreamWrapper {}
    unsafe impl Sync for StreamWrapper {}

    #[derive(Clone)]
    struct EventWrapper(pub *mut CUevent_st);
    unsafe impl Send for EventWrapper {}
    unsafe impl Sync for EventWrapper {}

    pub fn all_reduce_sync() {
        // Create NCCL unique ID.
        let id = cudarc::nccl::result::get_uniqueid().unwrap();
        println!("{:?}", id);
        let n_devices = 4;
        let count = 30;
        let mut slices = vec![];
        let mut streams = vec![];
        let mut events = vec![];

        let mut ctx_mutex: Vec<Arc<Mutex<usize>>> = vec![];

        cudarc::driver::result::init().unwrap();
        for i in 0..n_devices {
            ctx_mutex.push(Arc::new(Mutex::new(0)));
            // Create stream.
            let device_ptr = cudarc::driver::result::device::get(i as i32).unwrap();
            let _ctx = unsafe {
                let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                cudarc::driver::result::ctx::set_current(ctx).unwrap();
                ctx
            };
            let stream = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("Can create a new stream.");

            // Create the data and copy it on the GPU.
            let src = &vec![(i + 1) as f32 * 1.0; count];
            unsafe {
                let cu_device_ptr =
                    cudarc::driver::result::malloc_sync(src.len() * std::mem::size_of::<f32>())
                        .unwrap();
                cudarc::driver::result::memcpy_htod_async(cu_device_ptr, src, stream).unwrap();
                slices.push(cu_device_ptr);
            }
            streams.push(StreamWrapper(stream));

            let event = cudarc::driver::result::event::create(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
            )
            .unwrap();
            events.push(EventWrapper(event));
        }

        let mut handles = vec![];
        for i in 0..n_devices {
            // Spawn 1 thread per GPU.
            let slice_ptr = slices[i].clone();
            let stream = streams[i].clone();
            let tex = ctx_mutex[i].clone();
            let event = events[i].clone();
            let handle = std::thread::spawn(move || {
                let guard = tex.as_ref().lock().unwrap();
                // Create CUDA context and stream.
                let device_ptr = cudarc::driver::result::device::get(i as i32).unwrap();
                let _ctx = unsafe {
                    let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                    cudarc::driver::result::ctx::set_current(ctx).unwrap();
                    ctx
                };

                // Create communicator.
                let mut comm = MaybeUninit::uninit();
                let comm = unsafe {
                    cudarc::nccl::result::comm_init_rank(
                        comm.as_mut_ptr(),
                        n_devices as i32,
                        id,
                        i as i32,
                    )
                    .unwrap();
                    comm.assume_init()
                };

                println!("Comm");

                let stream2 = stream.clone().0;
                let event2 = event.clone().0;
                // Perform all_reduce.
                for _k in 0..1 {
                    unsafe {
                        cudarc::nccl::result::all_reduce(
                            slice_ptr as *const _,
                            slice_ptr as *mut _,
                            count,
                            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
                            cudarc::nccl::sys::ncclRedOp_t::ncclSum,
                            comm,
                            stream2 as _,
                        )
                        .unwrap();

                        cudarc::driver::result::event::record(event2, stream2).unwrap();
                    }
                    println!("All_reduce");
                }

                drop(guard);
                println!("Thread done.")
            });
            handles.push(handle);
        }

        // Wait for threads
        // for handle in handles {
        //     println!("Joining");
        //     handle.join().unwrap();
        // }
        // println!("Joined");

        // Transfer datat to CPU and print.
        for (i, slice) in slices.iter().enumerate() {
            let guard = ctx_mutex[i].as_ref().lock().unwrap();

            let stream = streams[i].0;
            unsafe {
                cudarc::driver::result::event::synchronize(events[i].0).unwrap();
                cudarc::driver::result::event::destroy(events[i].0).unwrap();
            }

            drop(guard);

            let mut dst: Vec<f32> = Vec::with_capacity(count);
            #[allow(clippy::uninit_vec)]
            unsafe {
                dst.set_len(count);
                cudarc::driver::result::memcpy_dtoh_async(&mut dst, *slice, stream).unwrap();
            };

            println!("my_vec {} : {:?}", i, dst);
        }

        // Wait for threads
        for handle in handles {
            println!("Joining");
            handle.join().unwrap();
        }
        println!("Joined");
    }

    pub fn all_reduce_in_place() {
        // Create NCCL unique ID.
        let id = cudarc::nccl::result::get_uniqueid().unwrap();
        println!("{:?}", id);
        let n_devices = 4;
        let count = 30;
        let mut slices = vec![];

        cudarc::driver::result::init().unwrap();
        for i in 0..n_devices {
            // Create stream.
            let device_ptr = cudarc::driver::result::device::get(i as i32).unwrap();
            let _ctx = unsafe {
                let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                cudarc::driver::result::ctx::set_current(ctx).unwrap();
                ctx
            };
            let stream = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("Can create a new stream.");

            // Create the data and copy it on the GPU.
            let src = &vec![(i + 1) as f32 * 1.0; count];
            unsafe {
                let cu_device_ptr =
                    cudarc::driver::result::malloc_sync(src.len() * std::mem::size_of::<f32>())
                        .unwrap();
                cudarc::driver::result::memcpy_htod_async(cu_device_ptr, src, stream).unwrap();
                slices.push(cu_device_ptr);
            }
        }

        let mut handles = vec![];
        for i in 0..n_devices {
            // Spawn 1 thread per GPU.
            let slice_ptr = slices[i].clone();
            let handle = std::thread::spawn(move || {
                // Create CUDA context and stream.
                let device_ptr = cudarc::driver::result::device::get(i as i32).unwrap();
                let _ctx = unsafe {
                    let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                    cudarc::driver::result::ctx::set_current(ctx).unwrap();
                    ctx
                };
                let stream = cudarc::driver::result::stream::create(
                    cudarc::driver::result::stream::StreamKind::NonBlocking,
                )
                .expect("Can create a new stream.");
                println!("Stream");

                // Create communicator.
                let mut comm = MaybeUninit::uninit();
                let comm = unsafe {
                    cudarc::nccl::result::comm_init_rank(
                        comm.as_mut_ptr(),
                        n_devices as i32,
                        id,
                        i as i32,
                    )
                    .unwrap();
                    comm.assume_init()
                };
                println!("Comm");

                // Perform all_reduce.
                for _k in 0..1 {
                    unsafe {
                        cudarc::nccl::result::all_reduce(
                            slice_ptr as *const _,
                            slice_ptr as *mut _,
                            count * 4,
                            cudarc::nccl::sys::ncclDataType_t::ncclFloat32,
                            cudarc::nccl::sys::ncclRedOp_t::ncclSum,
                            comm,
                            stream as _,
                        )
                        .unwrap();
                    }
                    println!("All_reduce");
                }
                println!("Thread done.")
            });
            handles.push(handle);
        }

        // Wait for threads
        for handle in handles {
            println!("Joining");
            handle.join().unwrap();
        }
        println!("Joined");

        // Transfer datat to CPU and print.
        for (i, slice) in slices.iter().enumerate() {
            let device_ptr = cudarc::driver::result::device::get(i as i32).unwrap();
            let _ctx = unsafe {
                let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
                cudarc::driver::result::ctx::set_current(ctx).unwrap();
                ctx
            };
            println!("Start");
            let stream = cudarc::driver::result::stream::create(
                cudarc::driver::result::stream::StreamKind::NonBlocking,
            )
            .expect("Can create a new stream.");

            let mut dst: Vec<f32> = Vec::with_capacity(count);
            #[allow(clippy::uninit_vec)]
            unsafe {
                dst.set_len(count);
                cudarc::driver::result::memcpy_dtoh_async(&mut dst, *slice, stream).unwrap();
            };

            println!("my_vec {} : {:?}", i, dst);
        }
    }

    pub fn all_reduce_in_place_old() {
        let id = cudarc::nccl::Id::new().unwrap();
        println!("{:?}", id);
        let n_devices = 4;
        let mut slices = vec![];

        for i in 0..n_devices {
            let context = CudaContext::new(i).unwrap();
            let stream = context.default_stream();
            let slice = stream.clone_htod(&vec![(i + 1) as f32 * 1.0; 30]).unwrap();
            slices.push(slice);
        }
        let mut handles = vec![];

        for i in 0..n_devices {
            let mut slice = slices[i].clone();
            let handle = std::thread::spawn(move || {
                let context = CudaContext::new(i).unwrap();
                let stream = context.default_stream();
                let res = Comm::from_rank(stream, i, n_devices, id);
                match res {
                    Ok(comm) => {
                        let res =
                            comm.all_reduce_in_place(&mut slice, &cudarc::nccl::ReduceOp::Sum);
                        match res {
                            Ok(_) => (),
                            Err(err) => {
                                let error = err.0;
                                println!("Error in all_reduce_in_place: {:?}", error);
                            }
                        }
                    }
                    Err(err) => println!("Error in from_rank: {:?}", err.0),
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        for (i, slice) in slices.iter().enumerate() {
            let context = CudaContext::new(i).unwrap();
            let stream = context.default_stream();
            let my_vec = stream.clone_dtoh(slice).unwrap();
            println!("my_vec {} : {:?}", i, my_vec);
        }
    }
}

pub use all_reduce_ex::*;

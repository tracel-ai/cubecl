#[cfg(not(target_family = "wasm"))]
mod _impl {
    use crate::compute::device_poison::PoisonWatch;
    use std::thread::JoinHandle;

    #[derive(Debug)]
    pub struct WgpuPoll {
        active_handle: std::sync::Arc<()>,
        cancel_sender: std::sync::mpsc::Sender<()>,
        poll_thread: JoinHandle<()>,
    }

    impl WgpuPoll {
        pub fn new(device: wgpu::Device, queue: wgpu::Queue, poison: PoisonWatch) -> Self {
            let active_handle = std::sync::Arc::new(());
            let thread_check = active_handle.clone();

            let (cancel_sender, cancel_receiver) = std::sync::mpsc::channel();
            let poll_thread = std::thread::spawn(move || {
                loop {
                    // Check whether the WgpuPoll, this thread, and something else is holding
                    // a handle. A lost device has nothing left to complete.
                    if std::sync::Arc::strong_count(&thread_check) > 2 && !poison.is_poisoned() {
                        if let Err(e) = poison.wait_unless_lost(&device, &queue, None)
                            && !poison.is_poisoned()
                        {
                            log::warn!("wgpu: waiting on the device failed during sync. ({e})")
                        }
                    } else {
                        // Do not cancel thread while someone still needs to poll.
                        if cancel_receiver.try_recv().is_ok() {
                            break;
                        }

                        std::thread::park();
                    }
                    std::thread::yield_now();
                }
            });

            Self {
                active_handle,
                cancel_sender,
                poll_thread,
            }
        }
        /// Get a handle, as long as it's alive the polling will be active.
        pub fn start_polling(&self) -> std::sync::Arc<()> {
            let handle = self.active_handle.clone();
            self.poll_thread.thread().unpark();
            handle
        }
    }

    /// How long a drop waits for the polling thread to let go of the device.
    const STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    impl Drop for WgpuPoll {
        /// Waits for the polling thread to stop, so the device and queue it holds are released
        /// before the server's drop returns. A server closed as the process exits is dropped
        /// just ahead of the driver's own exit teardown, which a thread still releasing the
        /// device would race.
        fn drop(&mut self) {
            self.cancel_sender
                .send(())
                .expect("Failed to shutdown polling thread.");
            self.poll_thread.thread().unpark();

            // A sync still in flight keeps the thread polling until it completes, so the wait
            // is bounded rather than a join.
            let start = std::time::Instant::now();
            while !self.poll_thread.is_finished() {
                if start.elapsed() >= STOP_TIMEOUT {
                    log::warn!(
                        "wgpu: the polling thread did not stop within {STOP_TIMEOUT:?}, leaving \
                         it to release the device on its own"
                    );
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

// On Wasm, the browser handles the polling loop, so we don't need anything.
#[cfg(target_family = "wasm")]
mod _impl {
    use crate::compute::device_poison::PoisonWatch;

    #[derive(Debug)]
    pub struct WgpuPoll {}
    impl WgpuPoll {
        pub fn new(_device: wgpu::Device, _queue: wgpu::Queue, _poison: PoisonWatch) -> Self {
            Self {}
        }
        pub fn start_polling(&self) -> alloc::sync::Arc<()> {
            alloc::sync::Arc::new(())
        }
    }
}

pub(crate) use _impl::*;

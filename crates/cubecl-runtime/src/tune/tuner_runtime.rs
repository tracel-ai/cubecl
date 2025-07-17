use core::{any::Any, pin::Pin};
use std::{
    sync::{OnceLock, mpsc::SyncSender},
    thread::JoinHandle,
};

use cubecl_common::stream_id::StreamId;

use crate::{channel::ComputeChannel, client::ComputeClient, server::ComputeServer};

static RUNTIME: OnceLock<TunerRuntime> = OnceLock::new();

trait ProfileLock: Send {
    fn aquire(&self) -> Option<StreamId>;
    fn release(&self, stream: Option<StreamId>);
}

struct DeviceProfileLock<S: ComputeServer, C: ComputeChannel<S>> {
    client: ComputeClient<S, C>,
}

impl<S: ComputeServer, C: ComputeChannel<S>> ProfileLock for DeviceProfileLock<S, C> {
    fn aquire(&self) -> Option<StreamId> {
        self.client.profile_acquire()
    }

    fn release(&self, stream: Option<StreamId>) {
        self.client.profile_release(stream, true)
    }
}

struct Message {
    fut: Pin<Box<dyn Future<Output = Box<dyn Any + Send>> + Send>>,
    lock: Box<dyn ProfileLock>,
    callback: SyncSender<Box<dyn Any + Send>>,
}

pub struct TunerRuntime {
    channel: SyncSender<Message>,
    _thread: JoinHandle<()>,
}

impl TunerRuntime {
    pub fn block_on<
        O: Send + 'static,
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    >(
        fut: impl Future<Output = O> + Send + 'static,
        client: ComputeClient<S, C>,
    ) -> O {
        let (callback, rec) = std::sync::mpsc::sync_channel::<Box<dyn Any + Send>>(1);
        let fut: Pin<Box<dyn Future<Output = Box<dyn Any + Send>> + Send>> = Box::pin(async move {
            let output = fut.await;
            let out: Box<dyn Any + Send> = Box::new(output);
            out
        });

        let msg = Message {
            fut,
            callback,
            lock: Box::new(DeviceProfileLock { client }),
        };

        let runtime = RUNTIME.get_or_init(Self::start);
        runtime.channel.send(msg).unwrap();

        if let Ok(val) = rec.recv() {
            *val.downcast().unwrap()
        } else {
            panic!()
        }
    }

    fn start() -> TunerRuntime {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message>(10);

        let thread = std::thread::spawn(move || {
            let current = StreamId::current();
            while let Ok(msg) = rec.recv() {
                let guard = msg.lock.aquire();
                println!("[{current}] Autotune aquired {guard:?}");
                let output = cubecl_common::future::block_on(msg.fut);
                msg.lock.release(guard);
                println!("[{current}] Autotune released {guard:?}");

                msg.callback.send(output).unwrap();
            }
        });

        TunerRuntime {
            channel: sender,
            _thread: thread,
        }
    }
}

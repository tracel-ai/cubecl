use std::{sync::mpsc::Sender, thread::spawn};

use cubecl_core::device::DeviceId;

use crate::compute::communication::{AllReduceArgs, CommunicationServer};

pub(crate) enum CommunicationMessage {
    Action(CommunicationAction),
    Close(),
}

pub(crate) enum CommunicationAction {
    AllReduce(AllReduceArgs),
    Sync(Sender<bool>),
}

#[derive(Clone, Debug)]
pub struct CommunicationClient {
    sender: Sender<CommunicationMessage>,
    // sync_fence: Arc<(Mutex<bool>, Condvar)>,
}
impl CommunicationClient {
    pub(crate) fn new(
        device_id: i32,
        all_ids: Vec<DeviceId>,
        stream: cudarc::nccl::sys::cudaStream_t,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = CommunicationServer::new(device_id, all_ids, stream);
        spawn(move || {
            loop {
                match rx.recv() {
                    Ok(msg) => match msg {
                        CommunicationMessage::Action(action) => server.process_message(action),
                        CommunicationMessage::Close() => break,
                    },
                    Err(_) => {
                        println!("Comm server hanged!");
                        break;
                    }
                }
            }
        });
        Self { sender: tx }
    }

    pub(crate) fn all_reduce(&self, all_reduce_args: AllReduceArgs) {
        self.sender
            .send(CommunicationMessage::Action(
                CommunicationAction::AllReduce(all_reduce_args),
            ))
            .unwrap();
    }

    pub(crate) fn is_finished(&self) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender
            .send(CommunicationMessage::Action(CommunicationAction::Sync(tx)))
            .unwrap();
        rx.recv().unwrap()
    }

    pub(crate) fn close(&self) {
        self.sender.send(CommunicationMessage::Close()).unwrap();
    }
}

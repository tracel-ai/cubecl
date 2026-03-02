use std::{sync::mpsc::Sender, thread::spawn};

use cubecl_core::device::DeviceId;

use crate::compute::communication::{AllReduceArgs, CommWrapper, CommunicationServer};

pub(crate) enum CommunicationMessage {
    Action(CommunicationAction),
    Close(),
}

pub(crate) enum CommunicationAction {
    AllReduce(AllReduceArgs),
    Comm(CommWrapper),
    HasComm(Sender<bool>),
}

#[derive(Clone, Debug)]
pub struct CommunicationClient {
    sender: Sender<CommunicationMessage>,
}
impl CommunicationClient {
    pub(crate) fn new(device_id: i32, all_ids: Vec<DeviceId>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = CommunicationServer::new(device_id, all_ids);
        println!("New server");
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
                    } // TODO : Find out why it disconnects.
                }
            }
        });
        println!("Spawned");
        Self { sender: tx }
    }

    pub(crate) fn all_reduce(&self, all_reduce_args: AllReduceArgs) {
        self.sender
            .send(CommunicationMessage::Action(
                CommunicationAction::AllReduce(all_reduce_args),
            ))
            .unwrap();
    }

    pub(crate) fn comm(&self, comm: *mut cudarc::nccl::sys::ncclComm) {
        self.sender
            .send(CommunicationMessage::Action(CommunicationAction::Comm(
                CommWrapper(comm),
            )))
            .unwrap();
    }

    pub(crate) fn has_comm(&self) -> bool {
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender
            .send(CommunicationMessage::Action(CommunicationAction::HasComm(
                tx,
            )))
            .unwrap();
        rx.recv().unwrap()
    }

    pub(crate) fn close(&self) {
        self.sender.send(CommunicationMessage::Close()).unwrap();
    }
}

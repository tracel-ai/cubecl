use crate::server::{
    BufferBinding, CommunicationId, CopyDescriptor, Handle, ReduceOperation, Server,
};
use alloc::{collections::VecDeque, sync::Arc, vec, vec::Vec};
use cubecl_common::{
    device::{DeviceId, ServiceId},
    device_handle::DeviceHandle,
};
use cubecl_environment::{collections::HashMap, stream::StreamId, sync::Mutex};
use cubecl_ir::ElemType;

static CLIENTS: Mutex<Option<HashMap<ServiceId, DeviceHandle<dyn Server>>>> = Mutex::new(None);
static SEQUENCERS: Mutex<Option<HashMap<CommunicationId, Arc<OperationSequencer>>>> =
    Mutex::new(None);

pub(crate) struct AllReduceRequest {
    pub(crate) device: DeviceHandle<dyn Server>,
    pub(crate) device_id: DeviceId,
    pub(crate) src: BufferBinding,
    pub(crate) dst: BufferBinding,
    pub(crate) dtype: ElemType,
    pub(crate) stream_id: StreamId,
    pub(crate) op: ReduceOperation,
    pub(crate) device_ids: Vec<DeviceId>,
}

pub(crate) struct TransferRequest {
    pub(crate) source: DeviceHandle<dyn Server>,
    pub(crate) destination_client: DeviceHandle<dyn Server>,
    pub(crate) source_descriptor: CopyDescriptor,
    pub(crate) destination_handle: Handle,
    pub(crate) dtype: ElemType,
    pub(crate) source_stream: StreamId,
    pub(crate) destination_stream: StreamId,
    pub(crate) source_device: DeviceId,
    pub(crate) destination_device: DeviceId,
}

pub(crate) struct OperationSequencer {
    id: CommunicationId,
    operations: Mutex<VecDeque<Operation>>,
}

enum Operation {
    AllReduce(AllReduce),
    Transfer(Transfer),
}

struct AllReduce {
    device_ids: Vec<DeviceId>,
    requests: Arc<Mutex<Vec<Option<AllReduceRequest>>>>,
    gate: Arc<Gate>,
}

struct Transfer {
    gate: Arc<Gate>,
}

#[cfg(feature = "std")]
struct Gate {
    open: std::sync::Mutex<bool>,
    ready: std::sync::Condvar,
}

#[cfg(feature = "std")]
impl Gate {
    fn new() -> Self {
        Self {
            open: std::sync::Mutex::new(false),
            ready: std::sync::Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.ready.wait(open).unwrap();
        }
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.ready.notify_all();
    }
}

#[cfg(not(feature = "std"))]
struct Gate;

#[cfg(not(feature = "std"))]
impl Gate {
    fn new() -> Self {
        Self
    }

    fn wait(&self) {}

    fn open(&self) {}
}

impl OperationSequencer {
    pub(crate) fn register_client(device: DeviceHandle<dyn Server>) {
        let mut clients = CLIENTS.lock();
        clients
            .get_or_insert_with(HashMap::new)
            .entry(device.service_id())
            .or_insert(device);
    }

    pub(crate) fn all_reduce(request: AllReduceRequest) {
        let device_ids = request.device_ids.clone();
        let devices = Self::devices(&request, &device_ids);
        Self::for_communicator(&device_ids, |sequencer| {
            sequencer.enqueue_all_reduce(request, devices);
        });
    }

    pub(crate) fn transfer(request: TransferRequest) {
        let device_ids = vec![
            request.source_device.clone(),
            request.destination_device.clone(),
        ];
        Self::for_communicator(&device_ids, |sequencer| {
            sequencer.enqueue_transfer(request);
        });
    }

    fn devices(
        request: &AllReduceRequest,
        device_ids: &[DeviceId],
    ) -> Vec<DeviceHandle<dyn Server>> {
        let clients = CLIENTS.lock();
        let clients = clients.as_ref().unwrap();
        device_ids
            .iter()
            .map(|device_id| {
                clients
                    .get(&ServiceId {
                        device: device_id.clone(),
                        service: request.device.service_id().service,
                    })
                    .expect("all collective clients must be loaded before all_reduce")
                    .clone()
            })
            .collect()
    }

    fn for_communicator(device_ids: &[DeviceId], action: impl FnOnce(&Self)) {
        let id = CommunicationId::from(device_ids.to_vec());
        let sequencer = {
            let mut sequencers = SEQUENCERS.lock();
            let sequencers = sequencers.get_or_insert_with(HashMap::new);
            sequencers
                .entry(id.clone())
                .or_insert_with(|| {
                    Arc::new(Self {
                        id,
                        operations: Mutex::new(VecDeque::new()),
                    })
                })
                .clone()
        };
        action(&sequencer);
    }

    fn enqueue_all_reduce(
        &self,
        request: AllReduceRequest,
        devices: Vec<DeviceHandle<dyn Server>>,
    ) {
        let mut operations = self.operations.lock();
        let index = operations.iter().position(|operation| match operation {
            Operation::AllReduce(all_reduce) => all_reduce.accepts(&request),
            Operation::Transfer(_) => false,
        });

        match index {
            Some(index) => match operations.get_mut(index).unwrap() {
                Operation::AllReduce(all_reduce) => all_reduce.push(request),
                Operation::Transfer(_) => unreachable!(),
            },
            None => operations.push_back(Operation::AllReduce(AllReduce::new(request, devices))),
        }
        self.advance(&mut operations);
    }

    fn enqueue_transfer(&self, request: TransferRequest) {
        let mut operations = self.operations.lock();
        operations.push_back(Operation::Transfer(Transfer::new(request)));
        self.advance(&mut operations);
    }

    fn advance(&self, operations: &mut VecDeque<Operation>) {
        loop {
            let queued_operations = operations.len();
            let Some(operation) = operations.front() else {
                break;
            };
            match operation {
                Operation::Transfer(_) => {
                    let Operation::Transfer(transfer) = operations.pop_front().unwrap() else {
                        unreachable!()
                    };
                    transfer.open();
                }
                Operation::AllReduce(all_reduce) => {
                    if !all_reduce.is_complete() {
                        if queued_operations > 1 {
                            log::error!(
                                "collective operation is waiting for ranks; communicator={:?} operation=all_reduce ranks_seen={:?}",
                                self.id,
                                all_reduce.ranks_seen()
                            );
                        }
                        break;
                    }
                    let Operation::AllReduce(all_reduce) = operations.pop_front().unwrap() else {
                        unreachable!()
                    };
                    all_reduce.open();
                }
            }
        }
    }
}

impl AllReduce {
    fn new(request: AllReduceRequest, mut devices: Vec<DeviceHandle<dyn Server>>) -> Self {
        let mut device_ids = request.device_ids.clone();
        device_ids.sort();
        devices.sort_by_key(|device| device.device_id());
        let requests = Arc::new(Mutex::new(
            (0..device_ids.len()).map(|_| None).collect::<Vec<_>>(),
        ));
        let gate = Arc::new(Gate::new());
        let all_reduce = Self {
            device_ids,
            requests,
            gate,
        };
        all_reduce.submit_waiters(devices);
        all_reduce.push(request);
        all_reduce
    }

    fn accepts(&self, request: &AllReduceRequest) -> bool {
        let mut device_ids = request.device_ids.clone();
        device_ids.sort();
        self.device_ids == device_ids
            && self
                .device_ids
                .iter()
                .position(|device_id| device_id == &request.device_id)
                .is_some_and(|index| self.requests.lock()[index].is_none())
    }

    fn push(&self, request: AllReduceRequest) {
        let index = self
            .device_ids
            .iter()
            .position(|device_id| device_id == &request.device_id)
            .unwrap();
        self.requests.lock()[index] = Some(request);
    }

    fn is_complete(&self) -> bool {
        self.requests.lock().iter().all(Option::is_some)
    }

    fn ranks_seen(&self) -> Vec<DeviceId> {
        self.requests
            .lock()
            .iter()
            .enumerate()
            .filter_map(|(index, request)| request.as_ref().map(|_| self.device_ids[index].clone()))
            .collect()
    }

    fn submit_waiters(&self, devices: Vec<DeviceHandle<dyn Server>>) {
        for (index, device) in devices.into_iter().enumerate() {
            let gate = self.gate.clone();
            let requests = self.requests.clone();
            device.submit(move |server| {
                gate.wait();
                let request = requests.lock()[index].take().unwrap();
                if let Err(err) = server.all_reduce(
                    request.src,
                    request.dst,
                    request.dtype,
                    request.stream_id,
                    request.op,
                    request.device_ids,
                ) {
                    log::error!("all_reduce failed; the destination carries the failure: {err}");
                }
            });
        }
    }

    fn open(self) {
        self.gate.open();
    }
}

impl Transfer {
    fn new(request: TransferRequest) -> Self {
        let TransferRequest {
            source,
            destination_client,
            source_descriptor,
            destination_handle,
            dtype,
            source_stream,
            destination_stream,
            source_device,
            destination_device,
        } = request;
        let gate = Arc::new(Gate::new());
        let source_gate = gate.clone();
        let source_destination_device = destination_device.clone();
        source.submit(move |server| {
            source_gate.wait();
            if let Err(err) = server.send(
                source_descriptor,
                dtype,
                source_stream,
                source_destination_device.clone(),
            ) {
                log::error!(
                    "send to {:?} failed; the peer's recv is left waiting: {err}",
                    source_destination_device
                );
            }
        });
        let destination_gate = gate.clone();
        destination_client.submit(move |server| {
            destination_gate.wait();
            if let Err(err) = server.recv(
                destination_handle,
                dtype,
                destination_stream,
                source_device.clone(),
            ) {
                log::error!(
                    "recv from {:?} failed; the destination carries the failure: {err}",
                    source_device
                );
                return;
            }
            if let Err(err) = server.sync_collective(destination_stream) {
                log::error!("sync_collective failed: {err}");
            }
        });
        source.flush_queue();
        destination_client.flush_queue();
        Self { gate }
    }

    fn open(self) {
        self.gate.open();
    }
}

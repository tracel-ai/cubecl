use crate::reuse::*;

use crossbeam::channel::{Receiver, Sender};
use cubecl_common::future::DynFut;
use cubecl_core::CubeCount;
use cubecl_core::server::{Binding, Bindings};
use cubecl_core::{ExecutionMode, MemoryUsage};
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::thread;

pub enum FarmOp {
    Sum,
    Product,
    Min,
    Max,
    Average,
}

pub enum DeviceCommand<FR: FarmRuntime> {
    Create {
        data: Vec<u8>,
        response: Sender<Handle>,
    },
    Execute {
        kernel: <<<FR as FarmRuntime>::R as Runtime>::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
    },
    Sync {
        response: Sender<()>,
    },
    Read {
        bindings: Vec<Binding>,
        response: Sender<Vec<Vec<u8>>>,
    },
    ReadOne {
        binding: Binding,
        response: Sender<Vec<u8>>,
    },
    MemoryUsage {
        response: Sender<MemoryUsage>,
    },
    Shutdown,
}

pub enum LinkCommand {
    AllReduce {
        handles: Vec<Handle>,
        operation: FarmOp,
        response: Sender<Result<()>>,
    },
    Broadcast {
        handle: Handle,
        root: usize,
        response: Sender<Result<()>>,
    },
    Barrier {
        response: Sender<Result<()>>,
    },
}
pub struct ThreadManager<FR: FarmRuntime> {
    pub unit_senders: HashMap<usize, Sender<DeviceCommand<FR>>>,
    pub group_senders: Option<HashMap<usize, Sender<LinkCommand>>>,
}

pub trait FarmChannel<FR: FarmRuntime + 'static>: Clone + core::fmt::Debug + Send + Sync {
    type LinkCommand;
    type UnitCommand;

    // Needed Methods
    fn create_copied(&self, farm: &Farm<FR>, group: usize, data: &[u8]) -> Result<Vec<Handle>>;

    fn create_explicit(
        &self,
        farm: &Farm<FR>,
        group: usize,
        data: Vec<&[u8]>,
    ) -> Result<Vec<Handle>>;

    fn create_unit(
        &self,
        farm: &Farm<FR>,
        group: usize,
        unit: usize,
        data: &[u8],
    ) -> Result<Handle>;

    fn read_group(
        &self,
        farm: &Farm<FR>,
        group: usize,
        bindings: Vec<Binding>,
    ) -> DynFut<Result<Vec<Vec<Vec<u8>>>>>;

    fn sync_group(&self, farm: &Farm<FR>, group: usize) -> DynFut<Result<()>>;

    fn execute_group(
        &self,
        farm: &Farm<FR>,
        group: usize,
        kernel: <<<FR as FarmRuntime>::R as Runtime>::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Vec<Binding>,
        mode: ExecutionMode,
    ) -> Result<()>;

    fn all_reduce(
        &self,
        farm: &Farm<FR>,
        group: usize,
        handles: &[Handle],
        operation: FarmOp,
    ) -> DynFut<Result<()>>;

    fn memory_usage_group(&self, farm: &Farm<FR>, group: usize) -> Vec<MemoryUsage>;

    // Provided methods
    fn setup_threads(farm: &Farm<FR>) -> Result<ThreadManager<FR>> {
        let mut unit_senders = HashMap::new();
        let mut group_senders = HashMap::new();

        for group in &farm.groups {
            let mut group_unit_senders = Vec::new();

            for unit in &group.units {
                let (tx, rx) = crossbeam::channel::unbounded();
                let device_index = unit.device_index();
                unit_senders.insert(device_index, tx.clone());
                group_unit_senders.push(tx.clone());
                let unit_clone = unit.clone();

                thread::spawn(move || {
                    Self::worker_comm(unit_clone, rx);
                });
            }

            if group.has_links() {
                let (tx, rx) = crossbeam::channel::unbounded();
                group_senders.insert(group.id, tx);
                let group_id = group.id;

                thread::spawn(move || {
                    Self::group_comm(group_id, rx, group_unit_senders);
                });
            }
        }

        Ok(ThreadManager {
            unit_senders,
            group_senders: if group_senders.is_empty() {
                None
            } else {
                Some(group_senders)
            },
        })
    }

    fn group_comm(
        group_id: usize,
        rx: Receiver<LinkCommand>,
        unit_senders: Vec<Sender<DeviceCommand<FR>>>,
    ) {
        loop {
            match rx.recv() {
                Ok(cmd) => match cmd {
                    LinkCommand::AllReduce {
                        handles,
                        operation,
                        response,
                    } => {
                        // TODO: Implement actual all-reduce logic with links
                        let _ = response.send(Ok(()));
                    }
                    LinkCommand::Broadcast {
                        handle,
                        root,
                        response,
                    } => {
                        // TODO: Implement broadcast logic with links
                        let _ = response.send(Ok(()));
                    }
                    LinkCommand::Barrier { response } => {
                        let mut sync_completed = true;
                        let mut sync_receivers = Vec::new();

                        for sender in &unit_senders {
                            let (tx, rx) = crossbeam::channel::bounded(1);
                            match sender.send(DeviceCommand::Sync { response: tx }) {
                                Ok(_) => sync_receivers.push(rx),
                                Err(_) => {
                                    sync_completed = false;
                                    break;
                                }
                            }
                        }

                        if sync_completed {
                            for rx in sync_receivers {
                                if rx.recv().is_err() {
                                    sync_completed = false;
                                    break;
                                }
                            }
                        }
                        let _ = response.send(if sync_completed {
                            Ok(())
                        } else {
                            Err(FarmError::BarrierError)
                        });
                    }
                },
                Err(_) => {
                    break;
                }
            }
        }
        for sender in unit_senders {
            let _ = sender.send(DeviceCommand::Shutdown);
        }
    }

    fn worker_comm(unit: FarmUnit<FR>, rx: Receiver<DeviceCommand<FR>>) {
        loop {
            match rx.recv() {
                Ok(cmd) => match cmd {
                    DeviceCommand::Create { data, response } => {
                        let handle = unit.client().create(data.as_slice());
                        let _ = response.send(handle);
                    }

                    DeviceCommand::Execute {
                        kernel,
                        count,
                        bindings,
                        mode,
                    } => unsafe {
                        unit.client().execute(kernel, count, bindings);
                    },

                    DeviceCommand::Sync { response } => {
                        unit.client().sync();
                        let _ = response.send(());
                    }

                    DeviceCommand::Read { bindings, response } => {
                        let result = unit.client().read(bindings);
                        let _ = response.send(result);
                    }

                    DeviceCommand::ReadOne { binding, response } => {
                        let data = unit.client().read_one(binding);
                        let _ = response.send(data);
                    }

                    DeviceCommand::MemoryUsage { response } => {
                        let usage = unit.client().memory_usage();
                        let _ = response.send(usage);
                    }

                    DeviceCommand::Shutdown => {
                        break;
                    }
                },

                Err(_) => {
                    break;
                }
            }
        }
    }
}

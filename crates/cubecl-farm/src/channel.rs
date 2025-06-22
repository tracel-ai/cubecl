use crate::reuse::*;

use crossbeam::channel::{Receiver, Sender, bounded};
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
    LinkOp {
        op: LinkCommand,
        data: ThreadCommand,
        response: Sender<Result<()>>,
    },
    Shutdown,
}

pub enum ThreadCommand {
    Target { target: Handle },
    RootTarget { root: Handle, target: Handle },
}

pub enum GroupCommand {
    Target { targets: Vec<Handle> },
    RootTarget { root: Handle, targets: Vec<Handle> },
}

pub enum LinkCommand {
    AllReduce,
    Broadcast,
    Reduce,
    AllGather,
    ReduceScatter,
    Barrier,
}

pub struct ThreadManager<FR: FarmRuntime> {
    pub unit_senders: HashMap<usize, Sender<DeviceCommand<FR>>>,
    pub group_senders: Option<HashMap<usize, Sender<(LinkCommand, GroupCommand)>>>,
}

impl<FR: FarmRuntime + 'static> ThreadManager<FR> {
    pub fn new(
        unit_senders: HashMap<usize, Sender<DeviceCommand<FR>>>,
        group_senders: Option<HashMap<usize, Sender<(LinkCommand, GroupCommand)>>>,
    ) -> Self {
        Self {
            unit_senders,
            group_senders,
        }
    }

    pub fn create_on_device(&self, device_index: usize, data: &[u8]) -> Result<Handle> {
        let (tx, rx) = bounded(1);
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::Create {
                data: data.to_vec(),
                response: tx,
            })
            .map_err(|_| FarmError::ChannelError)?;

        rx.recv().map_err(|_| FarmError::ChannelError)
    }

    pub fn create_on_group(
        &self,
        group_id: usize,
        devices: &[usize],
        data: &[u8],
    ) -> Result<Vec<Handle>> {
        let mut handles = Vec::new();

        for &device_index in devices {
            handles.push(self.create_on_device(device_index, data)?);
        }

        Ok(handles)
    }

    pub fn sync_device(&self, device_index: usize) -> Result<()> {
        let (tx, rx) = bounded(1);
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::Sync { response: tx })
            .map_err(|_| FarmError::ChannelError)?;

        rx.recv().map_err(|_| FarmError::ChannelError)
    }

    pub fn sync_devices(&self, devices: &[usize]) -> Result<()> {
        for &device_index in devices {
            self.sync_device(device_index)?;
        }
        Ok(())
    }

    pub fn execute_on_device(
        &self,
        device_index: usize,
        kernel: <<<FR as FarmRuntime>::R as Runtime>::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
    ) -> Result<()> {
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::Execute {
                kernel,
                count,
                bindings,
                mode,
            })
            .map_err(|_| FarmError::ChannelError)
    }

    pub fn read_from_device(&self, device_index: usize, binding: Binding) -> Result<Vec<u8>> {
        let (tx, rx) = bounded(1);
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::ReadOne {
                binding,
                response: tx,
            })
            .map_err(|_| FarmError::ChannelError)?;

        rx.recv().map_err(|_| FarmError::ChannelError)
    }

    pub fn memory_usage(&self, device_index: usize) -> Result<MemoryUsage> {
        let (tx, rx) = bounded(1);
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::MemoryUsage { response: tx })
            .map_err(|_| FarmError::ChannelError)?;

        rx.recv().map_err(|_| FarmError::ChannelError)
    }

    pub fn shutdown_device(&self, device_index: usize) -> Result<()> {
        let sender = self
            .unit_senders
            .get(&device_index)
            .ok_or(FarmError::InvalidDevice)?;

        sender
            .send(DeviceCommand::Shutdown)
            .map_err(|_| FarmError::ChannelError)
    }

    pub fn shutdown_all(&self) -> Result<()> {
        for sender in self.unit_senders.values() {
            let _ = sender.send(DeviceCommand::Shutdown);
        }
        Ok(())
    }
}

pub trait FarmChannel<FR: FarmRuntime + 'static>: Clone + core::fmt::Debug + Send + Sync {
    type LinkCommand;
    type UnitCommand;

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

    fn all_reduce(&self, farm: &Farm<FR>, group: usize, handles: &[Handle]) -> DynFut<Result<()>>;

    fn memory_usage_group(&self, farm: &Farm<FR>, group: usize) -> Vec<MemoryUsage>;

    // FIXME: We need to wait for a result more often

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
                let group_units = group.units.clone();

                thread::spawn(move || {
                    Self::group_comm(group_id, group_units, rx, group_unit_senders);
                });
            }
        }

        Ok(ThreadManager::new(
            unit_senders,
            if group_senders.is_empty() {
                None
            } else {
                Some(group_senders)
            },
        ))
    }

    fn group_comm(
        group_id: usize,
        group_units: Vec<FarmUnit<FR>>,
        rx: Receiver<(LinkCommand, GroupCommand)>,
        unit_senders: Vec<Sender<DeviceCommand<FR>>>,
    ) {
        loop {
            match rx.recv() {
                Ok((link_cmd, group_data)) => match (link_cmd, group_data) {
                    (LinkCommand::AllReduce, GroupCommand::Target { targets }) => {
                        for (i, (sender, target)) in
                            unit_senders.iter().zip(targets.iter()).enumerate()
                        {
                            let (tx, rx) = bounded(1);
                            let _ = sender.send(DeviceCommand::LinkOp {
                                op: LinkCommand::AllReduce,
                                data: ThreadCommand::Target {
                                    target: target.clone(),
                                },
                                response: tx,
                            });
                        }
                    }

                    (LinkCommand::Broadcast, GroupCommand::RootTarget { root, targets }) => {
                        for (i, (sender, target)) in
                            unit_senders.iter().zip(targets.iter()).enumerate()
                        {
                            let (tx, rx) = bounded(1);
                            let _ = sender.send(DeviceCommand::LinkOp {
                                op: LinkCommand::Broadcast,
                                data: ThreadCommand::RootTarget {
                                    root: root.clone(),
                                    target: target.clone(),
                                },
                                response: tx,
                            });
                        }
                    }

                    (LinkCommand::Reduce, GroupCommand::RootTarget { root, targets }) => {
                        for (i, (sender, target)) in
                            unit_senders.iter().zip(targets.iter()).enumerate()
                        {
                            let (tx, rx) = bounded(1);
                            let _ = sender.send(DeviceCommand::LinkOp {
                                op: LinkCommand::Reduce,
                                data: ThreadCommand::RootTarget {
                                    root: root.clone(),
                                    target: target.clone(),
                                },
                                response: tx,
                            });
                        }
                    }

                    (LinkCommand::AllGather, GroupCommand::Target { targets }) => {
                        for (i, (sender, target)) in
                            unit_senders.iter().zip(targets.iter()).enumerate()
                        {
                            let (tx, rx) = bounded(1);
                            let _ = sender.send(DeviceCommand::LinkOp {
                                op: LinkCommand::AllGather,
                                data: ThreadCommand::Target {
                                    target: target.clone(),
                                },
                                response: tx,
                            });
                        }
                    }

                    (LinkCommand::ReduceScatter, GroupCommand::Target { targets }) => {
                        for (i, (sender, target)) in
                            unit_senders.iter().zip(targets.iter()).enumerate()
                        {
                            let (tx, rx) = bounded(1);
                            let _ = sender.send(DeviceCommand::LinkOp {
                                op: LinkCommand::ReduceScatter,
                                data: ThreadCommand::Target {
                                    target: target.clone(),
                                },
                                response: tx,
                            });
                        }
                    }

                    (LinkCommand::Barrier, _) => {
                        let mut sync_completed = true;
                        let mut sync_receivers = Vec::new();

                        for sender in &unit_senders {
                            let (tx, rx) = bounded(1);
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
                    }

                    _ => {}
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

                    DeviceCommand::LinkOp { op, data, response } => {
                        let result = if let Some(link) = unit.link() {
                            match (op, data) {
                                (LinkCommand::AllReduce, ThreadCommand::Target { target }) => {
                                    link.all_reduce(target)
                                }

                                (
                                    LinkCommand::Broadcast,
                                    ThreadCommand::RootTarget { root, target },
                                ) => link.broadcast(root, target),

                                (
                                    LinkCommand::Reduce,
                                    ThreadCommand::RootTarget { root, target },
                                ) => {
                                    link.reduce(target, root);
                                    Ok(())
                                }

                                (LinkCommand::AllGather, ThreadCommand::Target { target }) => {
                                    link.all_gather(target);
                                    Ok(())
                                }

                                (LinkCommand::ReduceScatter, ThreadCommand::Target { target }) => {
                                    link.reduce_scatter(target);
                                    Ok(())
                                }

                                (LinkCommand::Barrier, _) => {
                                    unit.client().sync();
                                    Ok(())
                                }

                                _ => Err(FarmError::InvalidOperation),
                            }
                        } else {
                            Err(FarmError::NoLinksAvailable)
                        };
                        let _ = response.send(result);
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

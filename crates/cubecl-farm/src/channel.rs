use crate::reuse::*;

use crossbeam::channel::{Receiver, Sender};
use cubecl_common::future::DynFut;
use cubecl_core::CubeCount;
use cubecl_core::server::Binding;
use cubecl_core::{ExecutionMode, MemoryUsage};
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

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
        bindings: Vec<Binding>,
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
}

pub struct ThreadManager<FR: FarmRuntime> {
    pub unit_senders: HashMap<usize, Sender<DeviceCommand<FR>>>,
    pub group_senders: Option<HashMap<usize, Sender<LinkCommand>>>,
}

pub trait FarmChannel<FR: FarmRuntime>: Clone + core::fmt::Debug + Send + Sync {
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
        kernel: <<FR as FarmRuntime>::Server as ComputeServer>::Kernel,
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
        unimplemented!()
    }

    fn group_start(
        group_id: usize,
        rx: Receiver<LinkCommand>,
        unit_senders: Vec<Sender<DeviceCommand<FR>>>,
    ) {
        unimplemented!()
    }

    fn worker_start(unit: FarmUnit<FR>, rx: Receiver<DeviceCommand<FR>>) {
        unimplemented!()
    }
}

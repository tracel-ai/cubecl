/*!

Every Farm is composed out of FarmGroup's. In a FarmGroup we find a vector of FarmUnit. When a unit (a single GPU) is linked trough something like a NcclComm it has the state Linked. Otherwise it's seen as Single. With the given implementations we should be able get a good frame to iteraite over our Farm!

For now i will allign the trait FarmCube allong Nccl. The grouping of Nccl here is moved to Rust. This should make the concept more usable for Rust devs. The goal here will be to first just start each device in its own thread to then use a thread per group to manage all concurrent devices. With some helper functions we should be able to send kernels to all devices and so on. Later there will be cpu thread strategies implemented. Also posting to the CPU while having traing loops could be implemented further down the road.

*/

pub mod base;
pub mod error;

use cubecl_core::{Runtime, client::ComputeClient, server::Handle};
use error::*;
use std::collections::HashMap;

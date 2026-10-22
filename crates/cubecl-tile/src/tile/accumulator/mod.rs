//! Accumulators: opening ([`base`]), draining ([`drain`]), the shared-memory destinations
//! planes meet in, atomically ([`smem_accumulation`]), in cyclic rounds ([`smem_cyclic`]) or in a
//! slot each ([`smem_slots`]), and a cube's output as its planes reach it ([`planes_output`]).

pub(crate) mod accumulate_view;
pub(crate) mod base;
pub(crate) mod drain;
pub(crate) mod planes_output;
pub(crate) mod smem_accumulation;
pub(crate) mod smem_cyclic;
pub(crate) mod smem_slots;

pub(crate) use accumulate_view::*;
pub(crate) use base::*;
pub(crate) use drain::*;

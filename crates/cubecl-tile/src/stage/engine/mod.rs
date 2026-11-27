//! The engines a stage is filled by, one file each: what the engine states about itself
//! ([`Engine`](crate::launch::Engine)) and the move it makes.
//!
//! A new engine is a file here, its arm of [`Delivery`](crate::Delivery), and its arm wherever a
//! fill is handed to an engine: [`Delivery::engine`](crate::Delivery) for what it states,
//! [`move_lines`] for the lines its units move, and the slot and the copy that hand one a box.

pub(crate) mod async_per_unit;
pub(crate) mod base;
pub(crate) mod sync_per_unit;
pub(crate) mod tma;

pub(crate) use async_per_unit::{ASYNC_PER_UNIT, async_lines};
pub(crate) use base::{load_from, move_lines};
pub(crate) use sync_per_unit::{SYNC_PER_UNIT, sync_lines};
pub(crate) use tma::{TMA, issue_box, load_box};

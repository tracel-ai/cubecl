//! Nowhere to record into: every call is a no-op and nothing is ever stamped.

use super::{RecordEffect, RecordLevel, RecordsConfig, SessionId, Stamp, Stamped};
use alloc::string::String;
use core::time::Duration;

pub(crate) fn configure(_config: RecordsConfig) {}

pub(crate) fn level() -> RecordLevel {
    RecordLevel::Off
}

pub(crate) fn label(_label: String) {}

pub(crate) fn stamp() -> Option<Stamp> {
    None
}

pub(crate) fn offset(_session: SessionId) -> Option<Duration> {
    None
}

pub(crate) fn write<V: serde::Serialize>(
    _namespace: &str,
    _effect: RecordEffect,
    _stamped: &Stamped<V>,
) -> bool {
    false
}

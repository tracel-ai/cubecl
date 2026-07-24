use std::fmt::Display;

#[derive(Clone, Default)]
pub struct PlironEngine;

impl PlironEngine {
    pub(crate) fn run_kernel(&self, _pliron_data: &mut super::data::PlironData) {}
}

impl Display for PlironEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pliron representation will be put here")
    }
}

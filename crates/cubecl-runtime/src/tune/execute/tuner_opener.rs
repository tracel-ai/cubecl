use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hash::Hash;
use cubecl_environment::collections::HashMap;

use crate::tune::execute::tuner::Tuner;
use crate::tune::{AutotuneKeyable, TunableSet, TuneInputs};
use crate::{client::Client, device::HardwareIdentity};

/// The separator of the module path a local tuner is named by.
const MODULE_PATH_SEPARATOR: &str = "::";
/// What a table name joins the module path with in its place.
const TABLE_NAME_SEPARATOR: &str = "-";

/// Opens the tuners of a [`LocalTuner`](crate::tune::LocalTuner): one per hardware and candidate
/// list, which an id keeps for each list it executed.
pub struct TunerOpener<AK: AutotuneKeyable, ID> {
    /// An id's tuners, by id then by the checksum of the list each runs (Keyed Usage), made on the
    /// first execute.
    by_id: Option<TunersById<AK, ID>>,
    /// Every tuner, by the hardware and list it tunes (Keyed Usage), made on the first execute.
    opened: Option<HashMap<TunerIdentity, Arc<Tuner<AK>>>>,
}

/// The tuners of each id, by the checksum of the list each runs.
type TunersById<AK, ID> = HashMap<ID, HashMap<u64, Arc<Tuner<AK>>>>;

/// Which tuner a device's executes reach: one per hardware and candidate list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TunerIdentity {
    pub hardware: HardwareIdentity,
    /// The low half of the candidate list's [`compute_checksum`](TunableSet::compute_checksum):
    /// each device builds its own list, and one of the same hardware may build another.
    pub checksum: u64,
}

impl<AK: AutotuneKeyable, ID: Hash + Eq + Clone> TunerOpener<AK, ID> {
    pub const fn new() -> Self {
        Self {
            by_id: None,
            opened: None,
        }
    }

    /// The tuner `id` executes `operations` on, opened by the first id of its hardware to execute
    /// the same candidate list, its table named after `name`.
    pub fn open<I: TuneInputs, Out, Id>(
        &mut self,
        id: &ID,
        name: &str,
        client: &Client,
        operations: &TunableSet<AK, I, Out, Id>,
    ) -> Arc<Tuner<AK>> {
        let checksum = operations.checksum();
        if let Some(tuner) = self.opened(id, checksum) {
            return tuner.clone();
        }
        let identity = TunerIdentity {
            hardware: client.hardware_identity(),
            checksum,
        };
        let tuner = self
            .opened
            .get_or_insert_with(HashMap::new)
            .entry(identity)
            .or_insert_with_key(|identity| {
                Arc::new(Tuner::new(&table_name_of_path(name), &identity.hardware))
            })
            .clone();
        self.by_id
            .get_or_insert_with(HashMap::new)
            .entry(id.clone())
            .or_default()
            .insert(checksum, tuner.clone());
        tuner
    }

    /// The tuner `id` executed the list of `checksum` on, if it did.
    pub fn opened(&self, id: &ID, checksum: u64) -> Option<&Arc<Tuner<AK>>> {
        self.by_id.as_ref()?.get(id)?.get(&checksum)
    }

    /// Close every tuner, handing them back to be dropped where no lock is held: a captured
    /// value's drop may call back into the local tuner.
    pub fn close(&mut self) -> Vec<Arc<Tuner<AK>>> {
        self.by_id.iter_mut().for_each(HashMap::clear);
        self.opened
            .iter_mut()
            .flat_map(|opened| opened.drain().map(|(_, tuner)| tuner))
            .collect()
    }
}

/// The table name of a local tuner named after `module_path`.
fn table_name_of_path(module_path: &str) -> String {
    module_path.replace(MODULE_PATH_SEPARATOR, TABLE_NAME_SEPARATOR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_module_path_names_a_table_without_its_separators() {
        assert_eq!(
            table_name_of_path("crate::kernel-matmul"),
            "crate-kernel-matmul"
        );
    }

    #[test]
    fn an_opener_that_never_opened_has_no_tuner() {
        let opener = TunerOpener::<String, u8>::new();

        assert!(opener.opened(&0, 0).is_none());
    }
}

//! A prune of an environment where compiled kernels are kept between runs.

use super::Prune;
use super::kernels::KernelPruner;
use cubecl_environment::environment::{
    BuildStamp, EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned, RecordedUses,
    TunePruning, UsedContent,
};
use cubecl_environment::persistence::{Access, Database};
use cubecl_environment::records::Records;

impl Prune for EnvironmentId {
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError> {
        let database = self.database(Access::ReadWrite)?;
        let build = BuildStamp::current();
        let uses = RecordedUses::read(&database, build);
        let kernels = KernelPruner::new(build).prune(&database, &uses);
        uses.purge_other_builds(&database, &UsedContent::Kernels);
        let tunes = pruned_tunes(&database, &uses, options.tunes);
        let records = Records::new(&database).retain(options.records);
        Ok(Pruned {
            kernels,
            tunes,
            records,
        })
    }
}

/// Remove from `database` the tunes `tunes` says go, by what `uses`
/// records of this build's work, and tally them.
fn pruned_tunes(database: &Database, uses: &RecordedUses, tunes: TunePruning) -> PruneTally {
    match tunes {
        TunePruning::KeepAll => PruneTally::default(),
        TunePruning::KeepUsed => uses.prune_unused_tunes(database),
    }
}

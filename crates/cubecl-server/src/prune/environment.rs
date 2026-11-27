//! A prune of an environment where compiled kernels are kept between runs.

use super::Prune;
use super::kernels::KernelPruner;
use cubecl_environment::environment::{
    AnswerPruning, BuildStamp, EnvironmentError, EnvironmentId, PruneOptions, PruneTally, Pruned,
    RecordedUses, UsedContent,
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
        let answers = pruned_answers(&database, &uses, options.answers);
        let records = Records::new(&database).retain(options.records);
        Ok(Pruned {
            kernels,
            answers,
            records,
        })
    }
}

/// Remove from `database` the answers `answers` says go, by what `uses`
/// records of this build's work, and tally them.
fn pruned_answers(database: &Database, uses: &RecordedUses, answers: AnswerPruning) -> PruneTally {
    match answers {
        AnswerPruning::KeepAll => PruneTally::default(),
        AnswerPruning::KeepUsed => uses.prune_unused_answers(database),
    }
}

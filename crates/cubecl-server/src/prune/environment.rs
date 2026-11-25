//! A prune of an environment, where compiled kernels are kept between runs.

use super::Prune;
use super::answers::AnswerPruner;
use super::launches::WorkLaunches;
use cubecl_environment::environment::{
    BuildStamp, EnvironmentError, EnvironmentId, PruneOptions, Pruned,
};
use cubecl_environment::persistence::Access;
use cubecl_environment::records::Records;

impl Prune for EnvironmentId {
    fn prune(&self, options: PruneOptions) -> Result<Pruned, EnvironmentError> {
        let database = self.database(Access::ReadWrite)?;
        let build = BuildStamp::current();
        let kernels = WorkLaunches::read(&database, build).prune(&database);
        let answers = AnswerPruner::read(&database, build).prune(&database);
        let records = Records::new(&database).retain(options.records);
        Ok(Pruned {
            kernels,
            answers,
            records,
        })
    }
}

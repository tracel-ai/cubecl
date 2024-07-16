pub(crate) mod check;
pub(crate) mod ci;
pub(crate) mod pull_request_checks;
pub(crate) mod test;

use clap::ValueEnum;
use strum::{Display, EnumIter, EnumString};

#[derive(EnumString, EnumIter, Display, Clone, PartialEq, ValueEnum)]
#[strum(serialize_all = "lowercase")]
pub(crate) enum Target {
    All,
    Crates,
    Examples,
}

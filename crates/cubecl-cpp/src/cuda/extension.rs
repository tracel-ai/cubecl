use crate::Dialect;

use super::mma::{MmaCast, MmaExecute, MmaFill, MmaLoad, MmaStore};

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension<D: Dialect> {
    #[default]
    NoExtension,
    MmaSync(MmaSyncExtension<D>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MmaSyncExtension<D: Dialect> {
    Fill(MmaFill<D>),
    Load(MmaLoad<D>),
    Execute(MmaExecute<D>),
    Store(MmaStore<D>),
    Cast(MmaCast<D>),
}

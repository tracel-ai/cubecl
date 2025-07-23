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
impl<D: Dialect> MmaSyncExtension<D> {
    pub fn format_mma(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MmaSyncExtension::Fill(fill) => fill.format_extension(f),
            MmaSyncExtension::Load(load) => load.format_extension(f),
            MmaSyncExtension::Execute(execute) => execute.format_extension(f),
            MmaSyncExtension::Store(store) => store.format_extension(f),
            MmaSyncExtension::Cast(cast) => cast.format_extension(f),
        }
    }
}

use pliron::context::Context;

use super::mma::{WmmaCast, WmmaExecute, WmmaFill, WmmaLoad, WmmaStore};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension {
    #[default]
    NoExtension,
    Wmma(WmmaExtension),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WmmaExtension {
    Fill(WmmaFill),
    Load(WmmaLoad),
    Execute(WmmaExecute),
    Store(WmmaStore),
    Cast(WmmaCast),
}

impl WmmaExtension {
    pub fn format_wmma(
        &self,
        f: &mut core::fmt::Formatter<'_>,
        ctx: &Context,
    ) -> core::fmt::Result {
        match self {
            WmmaExtension::Fill(fill) => fill.format_extension(f, ctx),
            WmmaExtension::Load(load) => load.format_extension(f, ctx),
            WmmaExtension::Execute(execute) => execute.format_extension(f, ctx),
            WmmaExtension::Store(store) => store.format_extension(f, ctx),
            WmmaExtension::Cast(cast) => cast.format_extension(f, ctx),
        }
    }
}

use darling::{Error, FromMeta, ast::NestedMeta};
use proc_macro2::TokenStream;
use quote::quote;
use syn::Meta;

/// When a harness is compiled, from `harness` on `#[cube(..)]` or on a struct
/// deriving `CubeHarness`:
///
/// - `harness` is `harness(cfg(test))`
/// - `harness(cfg(<predicate>))` forwards the predicate verbatim as
///   `#[cfg(<predicate>)]`, so it's evaluated in the crate using the macro
/// - `harness(always)` is ungated
#[derive(Clone, Debug)]
pub enum HarnessGate {
    Cfg(TokenStream),
    Always,
}

impl HarnessGate {
    /// The attribute to put on every item of the harness.
    pub fn attr(&self) -> TokenStream {
        match self {
            HarnessGate::Cfg(predicate) => quote![#[cfg(#predicate)]],
            HarnessGate::Always => TokenStream::new(),
        }
    }
}

impl Default for HarnessGate {
    fn default() -> Self {
        HarnessGate::Cfg(quote![test])
    }
}

impl FromMeta for HarnessGate {
    fn from_word() -> darling::Result<Self> {
        Ok(HarnessGate::default())
    }

    fn from_list(items: &[NestedMeta]) -> darling::Result<Self> {
        let expected = "expected `harness`, `harness(cfg(..))` or `harness(always)`";
        let [item] = items else {
            return Err(Error::custom(expected));
        };
        match item {
            NestedMeta::Meta(Meta::Path(path)) if path.is_ident("always") => {
                Ok(HarnessGate::Always)
            }
            NestedMeta::Meta(Meta::List(list)) if list.path.is_ident("cfg") => {
                Ok(HarnessGate::Cfg(list.tokens.clone()))
            }
            item => Err(Error::custom(expected).with_span(item)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::kernel::{KernelArgs, from_tokens};

    fn gate(tokens: TokenStream) -> darling::Result<Option<String>> {
        let args: KernelArgs = from_tokens(tokens)?;
        Ok(args.harness.map(|gate| gate.attr().to_string()))
    }

    #[test]
    fn gates_are_parsed() {
        assert_eq!(gate(quote![launch]).unwrap(), None);
        assert_eq!(
            gate(quote![harness]).unwrap(),
            Some(quote![#[cfg(test)]].to_string())
        );
        assert_eq!(
            gate(quote![harness(cfg(any(test, feature = "export_tests")))]).unwrap(),
            Some(quote![#[cfg(any(test, feature = "export_tests"))]].to_string())
        );
        assert_eq!(gate(quote![harness(always)]).unwrap(), Some(String::new()));
    }

    #[test]
    fn unknown_gates_are_rejected() {
        assert!(gate(quote![harness(test)]).is_err());
        assert!(gate(quote![harness(none)]).is_err());
        assert!(gate(quote![harness()]).is_err());
        assert!(gate(quote![harness(cfg(test), always)]).is_err());
        assert!(gate(quote![harness = "test"]).is_err());
    }
}

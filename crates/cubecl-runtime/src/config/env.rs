/// The variant of `T` that the environment variable `name` names with its `serde` name, as in
/// `cubecl.toml`. `None` when the variable is unset or names no variant, as for [`env_bool`].
pub fn env_variant<T: serde::de::DeserializeOwned>(name: &str) -> Option<T> {
    variant(&std::env::var(name).ok()?)
}

/// The variant of `T` with the `serde` name `value`.
fn variant<T: serde::de::DeserializeOwned>(value: &str) -> Option<T> {
    use serde::de::{IntoDeserializer, value::Error};

    T::deserialize(IntoDeserializer::<Error>::into_deserializer(value)).ok()
}

/// A boolean environment variable, or `None` when it is unset or unreadable.
///
/// An unrecognized value is `None` rather than an error: the variable is an
/// override, so failing to parse it means leaving the configured value alone.
pub fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.as_str() {
        "true" | "1" | "on" => Some(true),
        "false" | "0" | "off" => Some(false),
        _ => None,
    }
}

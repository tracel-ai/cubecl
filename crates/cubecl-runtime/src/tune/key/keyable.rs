use alloc::string::String;
use core::fmt::{Debug, Display};
use core::hash::Hash;

/// What a value needs to key an autotune: to be hashed and compared in memory, displayed in the
/// logs, and, where tunes persist, serialized into the table.
///
/// # Assumptions
///
/// - Two keys are equal exactly when one winner serves both: a tune measures one problem and
///   keeps its answer for every equal key.
/// - A key serialized and read back is equal to the key it was written from.
#[cfg(serializable)]
pub trait AutotuneKeyable:
    Clone
    + Debug
    + PartialEq
    + Eq
    + Hash
    + Display
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Send
    + Sync
    + 'static
{
}

/// What a value needs to key an autotune: to be hashed and compared in memory, and displayed in
/// the logs.
///
/// # Assumptions
///
/// - Two keys are equal exactly when one winner serves both: a tune measures one problem and
///   keeps its answer for every equal key.
#[cfg(not(serializable))]
pub trait AutotuneKeyable:
    Clone + Debug + PartialEq + Eq + Hash + Display + Send + Sync + 'static
{
}

impl AutotuneKeyable for String {}

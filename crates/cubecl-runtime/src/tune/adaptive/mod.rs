//! The adaptive strategy, which only the native driver can run: round robin with elimination.

pub mod candidate_tracker;
pub mod first_pass;
pub mod patience;
pub mod sample_keeper;
pub mod scheduler;

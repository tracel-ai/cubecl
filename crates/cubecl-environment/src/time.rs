#[cfg(not(target_os = "none"))]
pub use web_time::{Duration, Instant};

#[cfg(target_os = "none")]
pub use embassy_time::{Duration, Instant};

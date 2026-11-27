/// What a tune does with its candidates while a stage of it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneStage {
    /// Each candidate runs with its launches discarded on the tuned stream, so the kernels it
    /// reaches queue to compile together.
    Gathering,
    /// The candidates run for real on the tuned stream: their warm-ups and their timed samples.
    Measuring,
}

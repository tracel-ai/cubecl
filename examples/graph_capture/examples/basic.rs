fn main() -> Result<(), Box<dyn core::error::Error>> {
    graph_capture::dispatch!(R => graph_capture::basic::<R>(&Default::default()));
    Ok(())
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    graph_capture::dispatch!(R => graph_capture::stateful::<R>(&Default::default()));
    Ok(())
}

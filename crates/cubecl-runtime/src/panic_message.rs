//! What a caught panic said.

use alloc::string::String;
use core::any::Any;

/// The message a panic carried in `payload`, or a placeholder for one that carried none.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| String::from(*message))
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| String::from("a panic with no message"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::string::ToString;

    #[test]
    fn a_static_message_is_read() {
        let payload: Box<dyn Any + Send> = Box::new("the block divides the vector");

        assert_eq!(panic_message(&*payload), "the block divides the vector");
    }

    #[test]
    fn a_formatted_message_is_read() {
        let payload: Box<dyn Any + Send> = Box::new("the block is 3".to_string());

        assert_eq!(panic_message(&*payload), "the block is 3");
    }

    #[test]
    fn a_payload_without_a_message_reads_as_such() {
        let payload: Box<dyn Any + Send> = Box::new(3_u8);

        assert_eq!(panic_message(&*payload), "a panic with no message");
    }
}

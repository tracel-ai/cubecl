use alloc::string::{String, ToString};

#[derive(new)]
#[doc(hidden)]
pub struct BytesFormat {
    bytes: u64,
}

impl core::fmt::Display for BytesFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let unit = 1000;

        if self.bytes < unit {
            f.write_fmt(format_args!("{} B", self.bytes))
        } else {
            let size = self.bytes as f64;
            let exp = match size.log(1000.0).floor() as usize {
                0 => 1,
                e => e,
            };
            let unit_prefix = "KMGTPEZY".as_bytes();
            f.write_fmt(format_args!(
                "{:.2} {}B",
                (size / unit.pow(exp as u32) as f64),
                unit_prefix[exp - 1] as char,
            ))
        }
    }
}

pub(super) fn bytes_format(bytes: u64) -> String {
    BytesFormat::new(bytes).to_string()
}

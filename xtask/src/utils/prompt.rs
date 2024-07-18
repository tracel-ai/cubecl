use std::io::{self, Write};

pub(crate) fn ask_once(prompt: &str) -> bool {
    print!("{}\nDo you want to proceed? (yes/no): ", prompt);
    io::stdout().flush().expect("stdout should be flushed");

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("should be able to read stdin line");
    input.trim().to_lowercase() == "yes" || input.trim().to_lowercase() == "y"
}

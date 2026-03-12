#[macro_export]
macro_rules! unexpanded {
    () => ({
        panic!("Unexpanded Cube functions should not be called. ");
    });
    ($msg:expr) => ({
        panic!($msg);
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!($fmt, $($arg)*);
    });
}

#[macro_export]
macro_rules! size {
    ($name: ident) => {
        $name
    };
}

#[macro_export]
macro_rules! define {
    ($name: ident) => {
        $name
    };
}

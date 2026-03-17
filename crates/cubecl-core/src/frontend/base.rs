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
macro_rules! expand_error {
    () => ({
        panic!("An error occurred during kernel expansion");
    });
    ($msg:expr) => ({
        panic!(concat!("An error occurred during kernel expansion:\n", $msg));
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!(concat!("An error occurred during kernel expansion:\n", $fmt), $($arg)*);
    });
}

#[macro_export]
macro_rules! expand_assert {
    ($cond:expr) => ({
        assert!($cond, "An error occurred during kernel expansion");
    });
    ($cond:expr, $msg:expr) => ({
        assert!($cond, concat!("An error occurred during kernel expansion:\n", $msg));
    });
    ($cond:expr, $fmt:expr, $($arg:tt)*) => ({
        assert!($cond, concat!("An error occurred during kernel expansion:\n", $fmt), $($arg)*);
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

fn main() {
    all_reduce::hello_world();
    // for i in 0..10 {
    // all_reduce::all_reduce_in_place();
    all_reduce::all_reduce_sync();
    // }
    // all_reduce::all_reduce_in_place_old();
}

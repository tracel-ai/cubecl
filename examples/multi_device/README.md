# multi_device

A collective timed on every device at once, each member on its own stream.

| example | measures |
| ------- | -------- |
| `all_reduce` | peer access per pair, then one all-reduce of a decode step's `[1, 2048]` hidden state: each member's time and the slowest |
| `timing_cost` | a decode-like pass of a kernel then an all-reduce, many times over, unobserved, under an observer timing kernels and under one timing kernels and communications: issue and step time per collective |
| `exposed_wait` | the same all-reduce with the last member held back by spinning kernels: each member's part and its compute stream's wait, timed on its device, then the same all-reduce waited for at a sync every member reaches after it ended |

```sh
cargo run --release -p multi_device --features cuda --example all_reduce
cargo run --release -p multi_device --features hip --example all_reduce
cargo run --release -p multi_device --features hip --example exposed_wait
cargo run --release -p multi_device --features hip --example timing_cost
```

# multi_device

What the link between devices delivers, and a collective timed on each member's device by the
execution observer.

| example | measures |
| ------- | -------- |
| `link_probe` | per ordered pair, `to_client` and a kernel loading from and storing to the peer's memory; then all-reduce, all-gather and reduce-scatter over each member count; each a curve from 8 KiB to 512 MiB with the time's relative standard deviation, cached per pair |
| `timing_cost` | a decode-like pass of a kernel then an all-reduce, many times over, unobserved, under an observer timing kernels and under one timing kernels and communications: issue and step time per collective |
| `exposed_wait` | the same all-reduce with the last member held back by spinning kernels: each member's part and its compute stream's wait, timed on its device, then the same all-reduce waited for at a sync every member reaches after it ended |

```sh
cargo run --release -p multi_device --features hip --example link_probe
cargo run --release -p multi_device --features hip --example exposed_wait
cargo run --release -p multi_device --features hip --example timing_cost
```

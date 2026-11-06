# multi_device

A collective timed on every device at once, each member on its own stream.

| example | measures |
| ------- | -------- |
| `all_reduce` | peer access per pair, then one all-reduce of a decode step's `[1, 2048]` hidden state: each member's time and the slowest |

```sh
cargo run --release -p multi_device --features cuda --example all_reduce
cargo run --release -p multi_device --features hip --example all_reduce
```

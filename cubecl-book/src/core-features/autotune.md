# Autotune

Autotuning drastically simplifies kernel selection by running small benchmarks at runtime to figure
out the best kernels with the best configurations to run on the current hardware; an essential
feature for portability. This feature combines gracefully with comptime to test the effect of
different comptime values on performance; sometimes it can be surprising!

Even if the benchmarks may add some overhead when running the application for the first time, the
information gets cached on the device and will be reused. It is usually a no-brainer trade-off for
throughput-oriented programs such as deep learning models. You can even ship the autotune cache with
your program, reducing cold start time when you have more control over the deployment target.

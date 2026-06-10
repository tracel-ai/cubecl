# Constants

CubeCL offers constants to make dealing with hardware runtime values easier. Instead of propagating values in config structs or as arguements. While these are compile time constants **DO NOT TRUST THE RUST ANALYZER VALUE** because the default at rustc compile time, not comptime is 2. This is because of a warning constant warning Rust has.

## Plane

GPUs have something called planes, this is apart of SIMT and why they are so much faster than CPUs. What you need to understand is that there are two constants provided to you to help with this. Because the constants are define with the proper values at comptime.

Plane's Dimensions, reports if the hardware supports 32 or 64 for example.
```rs
PLANE_DIM
```

Plane's position, reports a position from 0 to the 64 (If max is 64).
```rs
PLANE_POS
```


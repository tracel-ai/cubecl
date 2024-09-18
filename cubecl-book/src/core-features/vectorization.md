# Vectorization

High-performance kernels should rely on SIMD instructions whenever possible, but doing so can
quickly get pretty complicated! With CubeCL, you can specify the vectorization factor of each input
variable when launching a kernel. Inside the kernel code, you still use only one type, which is
dynamically vectorized and supports automatic broadcasting. The runtimes are able to compile kernels
and have all the necessary information to use the best instruction! However, since the algorithmic
behavior may depend on the vectorization factor, CubeCL allows you to access it directly in the
kernel when needed, without any performance loss, using the comptime system!
